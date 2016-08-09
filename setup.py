from glob import glob

data_files = glob('mcplates/data/plate_boundaries/*') +  glob('data/continents/*') + glob('data/*')

metadata = dict( name= 'mcplates',
                 version = 0.1,
                 description='Bayesian Monte Carlo analysis of plate kinematics',
                 url='',
                 author='Ian Rose',
                 author_email='ian.r.rose@gmail.com',
                 license='GPL',
                 long_description='',
                 include_package_data=True,
                 packages = ['mcplates'],
                 package_data = {'mcplates' : data_files }
               )

from setuptools import setup

setup ( **metadata )
