# Software Engineering Hygiene for Data Science

All production-ready data science code should read and feel no different from
any other high-quality codebase. The principles of reliable builds, testable
code, straightforward and distilled interfaces,and clearly documented design
decisions vastly improve the ease-of-use and maintenance of any code.


At a minmum, _any_ machine learning or otherwise data-interacting code should
have the following in its `git` repository:
- A descriptive, Markdown-formatted **`README.md`** that explains what the [code does and its purpose](https://www.makeareadme.com/).
- [Python **docstrings**](https://www.python.org/dev/peps/pep-0257/) for the most important classes, modules, and public functions. 
- A simple, repeatable process for **environment** creation (e.g. a [`conda create`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). and [`pip install -r`](https://pip.pypa.io/en/stable/reference/pip_install/)).
- Automated **tests** on core functionality (i.e. meaningful) that can be reliably executed (i.e. [`pytest`](https://docs.pytest.org/en/latest/contents.html)).
- The `master` branch in a **clean, working state** at all times. Feature branches are where development should occur, including exploring breaking changes.
- The **commit history** on `master` should be [clean](https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/best-practices), clear, and descriptive. Intermediate commits should never be merged into the main code branch.
- Have **continuous integration (CI)** setup. The [GitLab CI](https://docs.gitlab.com/ee/ci/) job _should fail_ if _any code fails to build_ or if _any dependency fails to download_ or if _any test fails to pass_.


Additionally, production-quality machine learning code should strive for:
- Automated code formatting using [`black`](https://github.com/ambv/black) and [`git` hooks](https://githooks.com/).
- Actively use [`coverage`](https://coverage.readthedocs.io/) to inspect test code coverage to keep the coverage percentage as _high as possible_.
- 100% code coverage via automated unit tests.
- If the modeling work is ultimately going into a deployed service (i.e. an [`ai-model-server`](https://gitlab.healthcareit.net/ArtificialIntelligence/Components/Incubator/ai-model-server)-using service), then ensure that the deployed model code has an [integration test](https://en.wikipedia.org/wiki/Integration_testing) that works both locally (via [`docker run`](https://docs.docker.com/engine/reference/run/)) and on the production platform (i.e. [AWS SageMaker](https://aws.amazon.com/sagemaker/)).
- Documentation on every function, class, and module.
- Automated documentation building using [`sphinx`](http://www.sphinx-doc.org/en/stable/) or another community-supported documentation standard.
- Prose-style documentation further describing the project's function and how different components interact with one another. Clear, direct technical writing documenting the project's intent, the data situation, the business impact and importance, as well as how the code architecture is designed is incredibly useful information for onboarding new scientists and engineeers.
