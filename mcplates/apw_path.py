import numpy as np
import pymc

from . import poles
from . import distributions
from . import rotations


class APWPath(object):

    def __init__(self, name, paleomagnetic_pole_list, n_euler_poles):
        for p in paleomagnetic_pole_list:
            assert (isinstance(p, poles.PaleomagneticPole))

        self._name = name
        self._poles = paleomagnetic_pole_list
        self.n_euler_rotations = n_euler_poles

        age_list = [p.age for p in self._poles]
        self._start_age = max(age_list)
        self._start_pole = self._poles[np.argmax(age_list)]

        self._pole_position_fn = APWPath.generate_pole_position_fn(
            n_euler_poles, self._start_age)
        self.dbname = self._name + '.pickle'
        self.model_vars = None
        self.mcmc = None

    @staticmethod
    def generate_pole_position_fn(n_euler_poles, start_age):

        def pole_position(start, age, tpw_pole_angle, tpw_rate, *args):
            if len(args) != max( (n_euler_poles * 3 - 1), 0):
                raise Exception("Unexpected number of euler poles/changepoints: expected %i, got %i"%(n_euler_poles*3-1, len(args)))

            # Parse the variable length arguments into euler poles and
            # changepoints
            lon_lats = args[0:n_euler_poles]
            rates = args[n_euler_poles:2 * n_euler_poles]
            changepoints = list(args[2 * n_euler_poles:])
            # append present day to make then the same length
            changepoints.append(0.0)
            changepoints = np.sort(changepoints)[::-1]
            euler_poles = [poles.EulerPole(ll[0], ll[1], r)
                           for ll, r in zip(lon_lats, rates)]

            # make a starting pole
            pole = poles.PaleomagneticPole(start[0], start[1], age=age)
            time = start_age

            # make a TPW pole
            test_1 = np.array([0.,0.,1.])
            test_2 = np.array([1.,0.,0.])
            if np.dot(pole._pole, test_1) > np.dot(pole._pole, test_2):
                great_circle_pole = np.cross(pole._pole, test_2)
            else:
                great_circle_pole = np.cross(pole._pole, test_1)
            lon, lat, _ = rotations.cartesian_to_spherical(great_circle_pole)
            TPW = poles.EulerPole(lon[0], lat[0], tpw_rate)
            TPW.rotate(pole, tpw_pole_angle)

            if n_euler_poles == 0:
                pole.rotate(TPW, TPW.rate * (time-age))
            else:
                for e, c in zip(euler_poles, changepoints):
                    # add tpw contribution
                    e2 = e.copy()
                    e2.add(TPW)
                    if age < c:
                        angle = e2.rate * (time - c)
                        pole.rotate(e2, angle)
                        time = c
                    else:
                        angle = e2.rate * (time - age)
                        pole.rotate(e2, angle)
                        break
            lon_lat = np.array([pole.longitude, pole.latitude])
            return lon_lat

        return pole_position

    def create_model(self, site_lon_lat=[0., 0.], watson_concentration=0., rate_scale=2.5, tpw_rate_scale = None):
        """
        Parameters
        ----------
        watson_concentration: Watson girdle distribution parameter associated
        with the distance of the Euler pole from APW path. Default parameter of
        0 correspond to a uniform distribution of a sphere.

        rate_scale: scale parameter for an exponential plate rates distribution (in Myr/deg).
        Default parameter of 2.5 is based on the best fitting scale parameter
        for the 14 largest plates in the NNR-MORVEL56 model. In an exponential distribution,
        the expectation value for the rate is given by 1/lambda, where lambda is the rate scale.

        tpw_rate_scale: scale parameter for an exponential TPW rate distribution (in Myr/deg).
        Defaults to None, which corresponds to running a model with no TPW.
        ----------

        Bayesian analysis requires that prior probability distributions are
        specified for each of the model parameters in the inverse problem.
        This function allows for the prior probability distribution of the Euler
        pole directions and the Euler pole magnitudes to be specified. The
        function also specifies the prior probability distribution of the start
        of the inversion utilizing the oldest pole in the pole list, the
        changepoint if there are multiple Eulers (as a uniform distribution
        between the oldest and youngest pole) and the age of the poles (either
        normal or uniform distribution as specified in the pole list).
        """
        assert rate_scale > 0.0, "rate_scale must be a positive number."
        assert tpw_rate_scale == None or tpw_rate_scale > 0.0
        assert watson_concentration <= 0.0, "Nonnegative Watson concentration parameters are not supported."
        if tpw_rate_scale is None:
            self.include_tpw = False
        else:
            self.include_tpw = True

        model_vars = []
        args = []

        start = distributions.VonMisesFisher('start',
                                             lon_lat=(
                                                 self._start_pole.longitude, self._start_pole.latitude),
                                             kappa=poles.kappa_from_two_sigma(self._start_pole.angular_error),
                                             value=(0.,0.), observed=False)

        model_vars.append(start)

        # Make TPW pole
        if self.include_tpw:
            tpw_pole_angle = pymc.Uniform('tpw_pole_angle',
                0., 360., value=0., observed=False)
            tpw_rate = pymc.Exponential('tpw_rate', tpw_rate_scale)
            model_vars.append(tpw_pole_angle)
            model_vars.append(tpw_rate)

        # Make Euler pole direction random variables
        for i in range(self.n_euler_rotations):
            euler = distributions.WatsonGirdle(
                'euler_' + str(i), lon_lat=site_lon_lat, kappa=watson_concentration,
                value=(0.,0.), observed=False)
            model_vars.append(euler)
            args.append(euler)

        # Make Euler pole rate random variables
        for i in range(self.n_euler_rotations):
            rate = pymc.Exponential('rate_' + str(i), rate_scale)
            model_vars.append(rate)
            args.append(rate)

        # Make changepoint random variables
        age_list = [p.age for p in self._poles]
        for i in range(self.n_euler_rotations - 1):
            changepoint = pymc.Uniform(
                'changepoint_' + str(i), min(age_list), max(age_list))
            model_vars.append(changepoint)
            args.append(changepoint)

        # Make observed random variables
        for i, p in enumerate(self._poles):
            if p.age_type == 'gaussian':
                pole_age = pymc.Normal(
                    'a_' + str(i), mu=p.age, tau=np.power(p.sigma_age, -2.))
            elif p.age_type == 'uniform':
                pole_age = pymc.Uniform(
                    'a_' + str(i), lower=p.sigma_age[0], upper=p.sigma_age[1])


            # Include TPW rate if it is part of model_vars
            if self.include_tpw:
                lon_lat = pymc.Lambda('ll_' + str(i),
                        lambda st=start, a=pole_age, tpw=tpw_pole_angle, r=tpw_rate, args=args:
                                  self._pole_position_fn(st, a, tpw, r, *args),
                                  dtype=np.float, trace=False, plot=False)
            # Otherwise use zero for TPW rate.
            else:
                lon_lat = pymc.Lambda('ll_' + str(i),
                        lambda st=start, a=pole_age, args=args:
                                  self._pole_position_fn(st, a, 0., 0., *args),
                                  dtype=np.float, trace=False, plot=False)

            observed_pole = distributions.VonMisesFisher('p_' + str(i),
                                                         lon_lat,
                                                         kappa=poles.kappa_from_two_sigma(
                                                             p.angular_error),
                                                         observed=True,
                                                         value=(p.longitude, p.latitude))
            model_vars.append(pole_age)
            model_vars.append(lon_lat)
            model_vars.append(observed_pole)

        self.model_vars = model_vars

    def sample_mcmc(self, nsample=10000):
        if self.model_vars is None:
            raise Exception("No model has been created")
        self.mcmc = pymc.MCMC(self.model_vars, db='pickle', dbname=self.dbname)
        self.find_MAP()
        self.mcmc.sample(nsample, int(nsample / 5), 1)
        self.mcmc.db.close()
        self.load_mcmc()

    def load_mcmc(self):
        self.mcmc = pymc.MCMC(self.model_vars, db='pickle', dbname=self.dbname)
        self.mcmc.db = pymc.database.pickle.load(self.dbname)

    def find_MAP(self):
        self.MAP = pymc.MAP(self.model_vars)
        self.MAP.fit()
        self.logp_at_max = self.MAP.logp_at_max
        return self.logp_at_max

    def tpw_poles(self):
        if self.mcmc.db is None:
            raise Exception("No database loaded")
        if self.include_tpw == False:
            return []

        tpw_pole_angle_samples = self.mcmc.db.trace('tpw_pole_angle')[:]
        start_samples = self.mcmc.db.trace('start')[:]
        tpw_pole_samples = np.empty_like(start_samples)
        index = 0
        for start, tpw_pole_angle in zip(start_samples, tpw_pole_angle_samples):
            test_1 = np.array([0.,0.,1.])
            test_2 = np.array([1.,0.,0.])
            pole = poles.Pole(start[0], start[1], 1.0)
            if np.dot(pole._pole, test_1) > np.dot(pole._pole, test_2):
                great_circle_pole = np.cross(pole._pole, test_2)
            else:
                great_circle_pole = np.cross(pole._pole, test_1)
            lon, lat, _ = rotations.cartesian_to_spherical(great_circle_pole)
            TPW = poles.Pole(lon[0], lat[0], 1.0)
            TPW.rotate(pole, tpw_pole_angle)
            tpw_pole_samples[index, :] = [TPW.longitude, TPW.latitude]
            index+=1

        return tpw_pole_samples

    def tpw_rates(self):
        if self.mcmc.db is None:
            raise Exception("No database loaded")
        if self.include_tpw == False:
            return []

        rate_samples = self.mcmc.db.trace('tpw_rate')[:]
        return rate_samples

    def euler_directions(self):
        if self.mcmc.db is None:
            raise Exception("No database loaded")
        direction_samples = []
        for i in range(self.n_euler_rotations):
            samples = self.mcmc.db.trace('euler_' + str(i))[:]
            samples[:,0] = rotations.clamp_longitude( samples[:,0])
            direction_samples.append(samples)
        return direction_samples

    def euler_rates(self):
        if self.mcmc.db is None:
            raise Exception("No database loaded")
        rate_samples = []
        for i in range(self.n_euler_rotations):
            rate_samples.append(self.mcmc.db.trace('rate_' + str(i))[:])
        return rate_samples

    def changepoints(self):
        if self.mcmc.db is None:
            raise Exception("No database loaded")
        changepoint_samples = []
        for i in range(self.n_euler_rotations - 1):
            changepoint_samples.append(
                self.mcmc.db.trace('changepoint_' + str(i))[:])
        return changepoint_samples

    def ages(self):
        if self.mcmc.db is None:
            raise Exception("No database loaded")
        age_samples = []
        for i in range(len(self._poles)):
            age_samples.append(self.mcmc.db.trace('a_' + str(i))[:])
        return age_samples

    def compute_synthetic_poles(self, n=100):

        assert n <= len(self.mcmc.db.trace('start')[
                        :]) and n >= 1, "Number of requested samples is not in allowable range"
        interval = max(1, int(len(self.mcmc.db.trace('start')[:]) / n))
        assert(interval > 0)

        n_poles = len(self._poles)
        lats = np.zeros((n, n_poles))
        lons = np.zeros((n, n_poles))
        ages = np.zeros((n, n_poles))

        index = 0
        for i in range(n):

            # begin args list with placeholder for age
            if self.include_tpw:
                args = [self.mcmc.db.trace('start')[index], 0.0, self.mcmc.db.trace('tpw_pole_angle')[index], self.mcmc.db.trace('tpw_rate')[index]]
            else:
                args = [self.mcmc.db.trace('start')[index], 0.0, 0.0, 0.0]

            # add the euler pole direction arguments
            for j in range(self.n_euler_rotations):
                euler = self.mcmc.db.trace('euler_' + str(j))[index]
                args.append(euler)

            # add the euler pole rate arguments
            for j in range(self.n_euler_rotations):
                rate = self.mcmc.db.trace('rate_' + str(j))[index]
                args.append(rate)

            # add the switchpoint arguments
            for j in range(self.n_euler_rotations - 1):
                changepoint = self.mcmc.db.trace('changepoint_' + str(j))[index]
                args.append(changepoint)

            for j, a in enumerate(self._poles):
                # put in the relevant age
                args[1] = self.mcmc.db.trace('a_' + str(j))[index]
                lon_lat = self._pole_position_fn(*args)
                lons[i, j] = lon_lat[0]
                lats[i, j] = lon_lat[1]
                ages[i, j] = args[1]

            index += interval

        return lons, lats, ages

    def compute_synthetic_paths(self, n=100):

        assert n <= len(self.mcmc.db.trace('start')[
                        :]) and n >= 1, "Number of requested samples is not in allowable range"
        interval = max(1, int(len(self.mcmc.db.trace('start')[:]) / n))
        assert(interval > 0)

        n_segments = 100
        pathlats = np.zeros((n, n_segments))
        pathlons = np.zeros((n, n_segments))
        age_list = [p.age for p in self._poles]
        ages = np.linspace(max(age_list), min(age_list), n_segments)

        index = 0
        for i in range(n):
            # begin args list with placeholder for age
            if self.include_tpw:
                args = [self.mcmc.db.trace('start')[index], 0.0, self.mcmc.db.trace('tpw_pole_angle')[index], self.mcmc.db.trace('tpw_rate')[index]]
            else:
                args = [self.mcmc.db.trace('start')[index], 0.0, 0.0, 0.0]

            # add the euler pole direction arguments
            for j in range(self.n_euler_rotations):
                euler = self.mcmc.db.trace('euler_' + str(j))[index]
                args.append(euler)

            # add the euler pole rate arguments
            for j in range(self.n_euler_rotations):
                rate = self.mcmc.db.trace('rate_' + str(j))[index]
                args.append(rate)

            # add the switchpoint arguments
            for j in range(self.n_euler_rotations - 1):
                changepoint = self.mcmc.db.trace('changepoint_' + str(j))[index]
                args.append(changepoint)

            for j, a in enumerate(ages):
                args[1] = a  # put in the relevant age
                lon_lat = self._pole_position_fn(*args)
                pathlons[i, j] = lon_lat[0]
                pathlats[i, j] = lon_lat[1]
            index += interval

        return pathlons, pathlats

    def compute_poles_on_path(self, ages, n_poles=100):
        """
        For a given suite of paths, return the positions predicted on the paths
        by the inversion for a given list of ages.

        Parameters
        ----------
        self : the paths object
        ages : list of ages along the path in Ma (e.g. [10,30,50])
        n_poles : number of paths to sample and the resultant number of poles that
            will be returned for a given age.

        Returns
        -------
        pathlons, pathlats: an array of pathlons and an array pathlats with one
            column for each age
        """
        assert n_poles <= len(self.mcmc.db.trace('start')[
                        :]) and n_poles >= 1, "Number of requested samples is not in allowable range"
        interval = max(1, int(len(self.mcmc.db.trace('start')[:]) / n_poles))
        assert(interval > 0)

        n_ages = len(ages)
        pathlats = np.zeros((n_poles, n_ages))
        pathlons = np.zeros((n_poles, n_ages))

        index = 0
        for i in range(n_poles):
            # begin args list with placeholder for age
            if self.include_tpw:
                args = [self.mcmc.db.trace('start')[index], 0.0, self.mcmc.db.trace('tpw_pole_angle')[index], self.mcmc.db.trace('tpw_rate')[index]]
            else:
                args = [self.mcmc.db.trace('start')[index], 0.0, 0.0, 0.0]

            # add the euler pole direction arguments
            for j in range(self.n_euler_rotations):
                euler = self.mcmc.db.trace('euler_' + str(j))[index]
                args.append(euler)

            # add the euler pole rate arguments
            for j in range(self.n_euler_rotations):
                rate = self.mcmc.db.trace('rate_' + str(j))[index]
                args.append(rate)

            # add the switchpoint arguments
            for j in range(self.n_euler_rotations - 1):
                changepoint = self.mcmc.db.trace('changepoint_' + str(j))[index]
                args.append(changepoint)

            for j, a in enumerate(ages):
                args[1] = a  # put in the relevant age
                lon_lat = self._pole_position_fn(*args)
                pathlons[i, j] = lon_lat[0]
                pathlats[i, j] = lon_lat[1]
            index += interval

        return pathlons, pathlats
