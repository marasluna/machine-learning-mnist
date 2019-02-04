# machine-learning-mnist
A set of HTTP web services that classify and test the MNIST dataset using machine learning 

# Deployment
```
# launch the virtual environment
$ pipenv shell
# deploy dev to AWS
$ zappa deploy dev
```
# Rollback
```
# launch the virtual environment
$ pipenv shell
# undeploy dev from AWS
$ zappa undeploy dev
```
