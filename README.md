# Neural Architecture Search

A program to optimize the neural architecture selection by intelligently evaluating a set of architectures.  Built with 
Tensorflow's [AdaNet](https://github.com/tensorflow/adanet).

## Instructions

### Dependencies

Create your environment from the home directory of the repo as follows:

`conda env create -f environment.yml`

once finished loading, simply activate the environment with:

`conda activate adanet`

### Running the program

After installing dependencies, simply run the driver file in terminal by passing in the repo's home directory:
 
`python WORKING_AUTOENSEMBLER_DRIVER.py`
