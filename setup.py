import setuptools
import subprocess


# Git version creation.
def get_git_version():
    cmd_branch    = 'git rev-parse --abbrev-ref HEAD'
    git_branch = subprocess.check_output(cmd_branch.split()).decode('utf-8').strip()

    cmd_shorthash = "git log --pretty=format:%H -n 1"
    git_shorthash = subprocess.check_output(cmd_shorthash.split()).decode('utf-8').strip()

    cmd_latesttag = 'git describe --tags --abbrev=0'
    try:
        git_latesttag = subprocess.check_output(cmd_latesttag.split()).decode('utf-8').strip()
    except:
        git_latesttag = ''

    cmd_revcount  = 'git rev-list --count ' + git_shorthash
    git_revcount = subprocess.check_output(cmd_revcount.split()).decode('utf-8').strip()

    return '[' + git_branch + ']' + git_latesttag + '-' + git_revcount + '(' + git_shorthash + ')'


# Get the current mss_dataserver version, author and description.
for line in open('lib/mss_dataserver/__init__.py').readlines():
    if (line.startswith('__version__')
            or line.startswith('__author__')
            or line.startswith('__authorEmail__')
            or line.startswith('__description__')
            or line.startswith('__license__')
            or line.startswith('__keywords__')
            or line.startswith('__website__')):
        exec(line.strip())

# Define the scripts to be processed.
scripts = ['scripts/mss_dataserver',]

# Get the version from the git repository and write it to the version file.
version_file = 'lib/mss_dataserver/version.py'
with open(version_file, 'w') as fid:
    fid.write('__git_version__ = "' + get_git_version() + '"')


setuptools.setup(name = 'mss_dataserver',
                 version           = __version__,
                 description       = __description__,
                 author            = __author__,
                 author_email      = __authorEmail__,
                 url               = __website__,
                 license           = __license__,
                 keywords          = __keywords__,
                 platforms         = 'any',
                 scripts           = scripts,
                 package_dir       = {'': 'lib'},
                 packages          = ['mss_dataserver',
                                      'mss_dataserver.geometry',
                                      'mss_dataserver.monitorclient'],
                 install_requires  = ['asyncio',
                                      'lxml',
                                      'numpy',
                                      'obspy',
                                      'pyproj',
                                      'websockets'])

