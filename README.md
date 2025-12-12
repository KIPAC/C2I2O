# C2I2O
Cosmologies to Intermediates to Observables and Back Again


## Sample data for CDU Hackathon: Human Meets AI in Scientific Research Replication

You can find both input comsological parameters "cosmo.hdf5" and the resulting intermediate data products "intemeds.hdf5" here:

https://s3df.slac.stanford.edu/people/echarles/c2i2o/

## Creating more sample data

git clone git@github.com:KIPAC/C2I2O.git  # Or whichever git clone method you use
cd C2I2O           
pip install -e .   # install in editable mode (in case you want changes)
hash -r            # so that shell pick ups the command line tool 
c2i2o-c2i generate --cosmo-parameter-file cosmo.hdf5 --intermediates-file intermeds.hdf5 nb/c2i.yaml 

