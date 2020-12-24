
from setuptools import setup, find_packages

package_name = 'nas_codefest'


def parse_requirements(file):
    with open(file, "r") as fs:
        return [r for r in fs.read().splitlines() if
                (len(r.strip()) > 0 and not r.strip().startswith("#") and not r.strip().startswith("--"))]


requirements = parse_requirements('requirements.txt')

setup(name=package_name,
      version='1.0',
      license='Change HealthCare',
      description='Turing Pipeline',
      author='ArtificialIntelligence',
      author_email='',
      url='',
      packages=find_packages(exclude=['tests', 'scripts']),
      install_requires=requirements,
      include_package_data=True,
      dependency_links=[''],
      zip_safe=False)
