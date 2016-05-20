import numpy as np
import pymc

from . import poles
from . import distributions


class APWPath(object):

    def  __init__( self, name, paleomagnetic_pole_list, n_euler_poles ):
        for p in paleomagnetic_pole_list:
            assert (isinstance(p, poles.PaleomagneticPole))

        self._name = name
        self._poles = paleomagnetic_pole_list
        self._n_euler_poles = n_euler_poles

        age_list = [p.age for p in self._poles]
        self._start_age = max(age_list)
        self._start_pole = self._poles[np.argmax(age_list)]
        
 
        self._pole_position_fn = self._generate_pole_position_fn( n_euler_poles, self._start_age )
        self.dbname = self._name + '.pickle'
        self._model = None
        self.db = None

    def _generate_pole_position_fn( self, n_euler_poles, start_age ):

        def pole_position( start, age, *args ):
            if len(args) != (n_euler_poles * 3 - 1):
                raise Exception("Unexpected number of euler poles/changpoints")

            # Parse the variable length arguments into euler poles and changepoints
            lon_lats = args[0:n_euler_poles]
            rates = args[n_euler_poles:2*n_euler_poles]
            changepoints = list(args[2*n_euler_poles:])
            changepoints.append(0.0) #append present day to make then the same length
            changepoints =np.sort(changepoints)[::-1]
            euler_poles = [ poles.EulerPole( ll[0], ll[1], r) for ll,r in zip(lon_lats,rates)]

            #Now make a starting pole
            pole = poles.PaleomagneticPole(start[0], start[1], age=age)
            time = start_age

            for e,c in zip(euler_poles, changepoints):
                if age < c:
                    angle = e.rate * ( time - c )
                    pole.rotate(e, angle )
                    time = c
                else:
                    angle = e.rate * ( time - age )
                    pole.rotate(e, angle )
                    break
            lon_lat = np.array([pole.longitude, pole.latitude])
            return lon_lat

        return pole_position

    def create_model(self, site_lon_lat=[0.,0.], watson_concentration=0.):
        model_vars = []
        args = []

        start = distributions.VonMisesFisher('start', \
                                             lon_lat=(self._start_pole.longitude, self._start_pole.latitude),\
                                             kappa=poles.kappa_from_two_sigma(self._start_pole.angular_error))

        model_vars.append(start)

        # Make Euler pole direction random variables
        for i in range(self._n_euler_poles):
            euler = distributions.WatsonGirdle('euler_'+str(i), lon_lat=site_lon_lat, kappa=watson_concentration)
            model_vars.append(euler)
            args.append(euler)

        # Make Euler pole rate random variables
        for i in range(self._n_euler_poles):
            rate = pymc.Exponential('rate_'+str(i), 1.) 
            model_vars.append(rate)
            args.append(rate)

        # Make changepoint random variables
        age_list = [p.age for p in self._poles]
        for i in range(self._n_euler_poles-1):
            changepoint = pymc.Uniform('changepoint_'+str(i), min(age_list), max(age_list))
            model_vars.append(changepoint)
            args.append(changepoint)

        # Make observed random variables
        for i,p in enumerate(self._poles):
            if p.age_type == 'gaussian':
                pole_age = pymc.Normal('a_'+str(i), mu=p.age, tau=np.power(p.sigma_age, -2.))
            elif p.age_type == 'uniform':
                pole_age = pymc.Uniform('a_'+str(i), lower=p.sigma_age[0], upper=p.sigma_age[1])

            lon_lat = pymc.Lambda('ll_'+str(i), lambda st=start, a=pole_age, args=args : \
                                                      self._pole_position_fn(st, a, *args ),\
                                                      dtype=np.float, trace=False, plot=False)
            observed_pole = distributions.VonMisesFisher('p_'+str(i),\
                                                         lon_lat,\
                                                         kappa = poles.kappa_from_two_sigma(p.angular_error),\
                                                         observed=True,\
                                                         value=(p.longitude,p.latitude))
            model_vars.append(pole_age)
            model_vars.append(lon_lat)
            model_vars.append(observed_pole)

        self._model = pymc.Model( model_vars )

    def sample_mcmc( self, nsample=10000 ):
        if self._model is None:
           raise Exception("No model has been created")
        mcmc = pymc.MCMC(self._model, db='pickle', dbname=self.dbname)
        pymc.MAP(self._model).fit()
        mcmc.sample(nsample, int(nsample/5), 1)
        mcmc.db.close()
        self.load_mcmc()

    def load_mcmc(self):
        self.db = pymc.database.pickle.load(self.dbname)

    def euler_directions(self):
        if self.db is None:
            raise Exception("No database loaded")
        direction_samples = []
        for i in range(self._n_euler_poles):
            direction_samples.append(self.db.trace('euler_'+str(i))[:])
        return direction_samples

    def euler_rates(self):
        if self.db is None:
            raise Exception("No database loaded")
        rate_samples = []
        for i in range(self._n_euler_poles):
            rate_samples.append(self.db.trace('rate_'+str(i))[:])
        return rate_samples

    def changepoints(self):
        if self.db is None:
            raise Exception("No database loaded")
        changepoint_samples = []
        for i in range(self._n_euler_poles-1):
            changepoint_samples.append(self.db.trace('changepoint_'+str(i))[:])
        return changepoint_samples

    def ages(self):
        if self.db is None:
            raise Exception("No database loaded")
        age_samples = []
        for i in range(len(self._poles)):
            age_samples.append(self.db.trace('a_'+str(i))[:])
        return age_samples

    def compute_synthetic_poles(self, n = 100):

        assert n <= len(self.db.trace('rate_0')[:]) and n >= 1, "Number of requested samples is not in allowable range"
        interval = max(1, int(len( self.db.trace('rate_0')[:])/n))
        assert(interval > 0)

        n_poles = len(self._poles)
        lats = np.zeros( (n, n_poles) )
        lons = np.zeros( (n, n_poles) )
        ages = np.zeros( (n, n_poles) )

        index = 0
        for i in range(n):

            #begin args list with placeholder for age
            args = [self.db.trace('start')[index], 0.0]

            #add the euler pole direction arguments
            for j in range(self._n_euler_poles):
                euler = self.db.trace('euler_'+str(j))[index]
                args.append(euler)

            #add the euler pole rate arguments
            for j in range(self._n_euler_poles):
                rate = self.db.trace('rate_'+str(j))[index]
                args.append(rate)

            #add the switchpoint arguments
            for j in range(self._n_euler_poles-1):
                changepoint = self.db.trace('changepoint_'+str(j))[index]
                args.append(changepoint)

            for j,a in enumerate(self._poles):
                args[1] = self.db.trace('a_'+str(j))[index]  #put in the relevant age
                lon_lat = self._pole_position_fn( *args )
                lons[i,j] = lon_lat[0]
                lats[i,j] = lon_lat[1]
                ages[i,j] = args[1]

            index += interval

        return lons, lats, ages

    def compute_synthetic_paths(self, n=100):

        assert n <= len(self.db.trace('rate_0')[:]) and n >= 1, "Number of requested samples is not in allowable range"
        interval = max(1, int(len( self.db.trace('rate_0')[:])/n))
        assert(interval > 0)

        n_segments = 100
        pathlats = np.zeros( (n, n_segments) )
        pathlons = np.zeros( (n, n_segments) )
        age_list = [p.age for p in self._poles]
        ages = np.linspace(max(age_list), min(age_list), n_segments)

        index = 0
        for i in range(n):
            #begin args list with placeholder for age
            args = [self.db.trace('start')[index], 0.0]

            #add the euler pole direction arguments
            for j in range(self._n_euler_poles):
                euler = self.db.trace('euler_'+str(j))[index]
                args.append(euler)

            #add the euler pole rate arguments
            for j in range(self._n_euler_poles):
                rate = self.db.trace('rate_'+str(j))[index]
                args.append(rate)

            #add the switchpoint arguments
            for j in range(self._n_euler_poles-1):
                changepoint = self.db.trace('changepoint_'+str(j))[index]
                args.append(changepoint)

            for j,a in enumerate(ages):
                args[1] = a #put in the relevant age
                lon_lat = self._pole_position_fn( *args )
                pathlons[i,j] = lon_lat[0]
                pathlats[i,j] = lon_lat[1]
            index += interval

        return pathlons, pathlats


    if __name__ == "__main__":
        import os 
        if os.path.isfile(path.dbname):
            path.load_mcmc()
        else:
            path.sample_mcmc(10000)
        plot_trace(path.db.trace)
