# Developer Tools Installation

To develop and run a project's code, it is cruical to have several tools and 
system libraries installed. This document provides instructions on configuring
one's development environment that would be applicable to nearly every project from Change Healthcare's AI team.

1. OS X Developer Tools
2. Homebrew
3. SDKMAN
4. SSH Shared Library
5. Java SDK
6. Spark
7. Docker



## 1. Have the OS X Developer Tools Installed
The OS X developer tools are necessary to perform nearly any kind of
developmenton an OS X system. If you've worked on another project already,
chances are you've done this step. If not, then there are a few ways to
install:

1. Download XCode from the [Apple Developer](https://developer.apple.com/xcode/) website. Note that this requires an Apple ID.
2. Use the Change Healthcare "Self Service" application (pre-installed) to install XCode.
3. Prompt your system to install the developer tools by attempting to run a program that needs them (e.g. `make`). If not installed, then executing `make`in a terminal will open a message dialog that will assist you in installing the tools.

Note that options (1) and (2) install Xcode along with the developer tools.
Option (3) installs the required tools only. 



## 2. Homebrew as a System Library Package Manager
Homebrew is a widely used package manager for OS X. Often, we will use
Homebrew to install developer or system libraries and common tools missing
from the standard OS X installation (e.g. `wget`). The
[homebrew project website](https://brew.sh/) gives more details. To install,
execute:
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```



## 3. Use SDKMAN for Managing Multiple SDK Versions
SDKMAN is a different cross-platform package manger that is more focused on
allowing users to install & manage multiple versions of the same executable
tool or toolset. The [SDKMAN project website](https://sdkman.io/) gives more
details. To install, execute:
```bash
curl -s "https://get.sdkman.io" | bash
```



## 4. Insall `libssh2`
Ensure that `libssh2` is installed. Several tools use `ssh` and need access to
it as a shared library. To install `libssh2`, use homebrew:
```bash
brew install libssh2
```

##### Debugging: No `libssh2` for `emr-utils` Install
If you do not have `libssh2` on your system, then you will encounter a
compilation error `pip` is attempting to install the `emr-utils` dependency.
Specifically, in this installation, the `parallel-ssh` wheel will be built.
When `libssh2` is not on the system, you will see this error: 
```
  pssh/native/_ssh2.c:530:10: fatal error: 'libssh2.h' file not found
  #include "libssh2.h"
           ^~~~~~~~~~~
  1 error generated.
  error: command 'gcc' failed with exit status 1
  ----------------------------------------
  Failed building wheel for parallel-ssh
```



# 5. Ensure that `java` is installed
We need to ensure that we have JDK 1.8 installed and available on our system. Many programs run on the JVM (e.g. Spark) This means we need `java`, `javac`,
etc. Using SDKMAN, we can install a specific JDK version as:
```bash
sdk install java 8.0.201-oracle
```
This above command will install the Oracle 1.8 JVM at patch version `201`.



# 6. Installing Spark Versions
Several CHC AI Team projects use Spark. Importantly, it is crucial to have the
compatible version installed or otherwise active while working in a
Spark-enabled project.

If you needed to install version `2.2.1` (and knew this by looking at the
project's `requirements.txt` file and noting the `pyspark` version), then 
you would execute the following SDKMAN command:
```bash
sdk install spark 2.2.1
```

If you needed to switch to Spark `1.4.1`, you'd execute `sdk use spark 1.4.1`. 
NOTE that only one Spark version may be activate at a time for your entire 
system when using `pyspark`.

##### Debugging: Spark not installed
If you do not have Spark installed then, when executing `pytest`, you will 
encounter the following error: 
```
ValueError: Couldn't find Spark, make sure SPARK_HOME env is set or Spark is in an expected location (e.g. from homebrew installation).
```

##### Debugging: Wrong Spark version installed
If you have Spark installed, but it is the wrong version (e.g. `2.4.0` when the  project requires `2.2.0`), then you will observe weird behavior when
attempting to run tests using `pyspark`. 

The behavior that has been encountered w/ a `2.4.0` install is  `pytest`
will hang during its test-collection phase (before test execution):
```
$ pytest
=========================================================================================== test session starts ============================================================================================
platform darwin -- Python 3.6.8, pytest-3.3.0, py-1.7.0, pluggy-0.6.0
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/Users/.../ros_model_experiments/.hypothesis/examples')
rootdir: /Users/.../ros_model_experiments, inifile:
plugins: cov-2.5.1, hypothesis-4.4.3
collecting 0 items
```

Running with `--fulltrace` and killing shows that the process is stuck in the
bowels of `pyspark`. Specifically, it is waiting on establishing a network 
connection to the locally installed Spark driver that was located & started by
`findspark`:
```
$ pytest --fulltrace
...

            gateway_port = None
            # We use select() here in order to avoid blocking indefinitely if the subprocess dies
            # before connecting
            while gateway_port is None and proc.poll() is None:
                timeout = 1  # (seconds)
>               readable, _, _ = select.select([callback_socket], [], [], timeout)
E               KeyboardInterrupt

.../env/miniconda3/envs/ros_model_experiments/lib/python3.6/site-packages/pyspark/java_gateway.py:87: KeyboardInterrupt
```



## 7. Docker for Deployment
Every CHC AI Team service, when deployed, executes within the context of a
Docker image. This design enables one to run and test services locally via `docker run -p 8080:8080` (or `docker-compose up`, depending on the project).

To install `docker`, follow the [installation instructions](https://docs.docker.com/docker-for-mac/install/).
Importantly, after installation, be sure to `docker login` with your Docker 
Hub credentials.

