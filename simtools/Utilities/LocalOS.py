import platform
import getpass
from simtools.Utilities import distro


class LocalOS(object):
    """
    A Central class for representing values whose proper access methods may differ between platforms.
    """

    def get_linux_distribution():
        name = distro.name().lower()

        dist = ''
        if 'centos' in name:
            dist = 'CentOS'
        elif 'ubuntu' in name:
            dist = 'Ubuntu'
        elif 'debian' in name:
            dist = 'Debian'
        elif 'fedora' in name:
            dist = 'Fedora'

        return dist

    class UnknownOS(Exception):
        pass

    WINDOWS = 'win'
    LINUX = 'lin'
    MAC = 'mac'
    ALL = (WINDOWS, LINUX, MAC)
    OPERATING_SYSTEMS = {
        'windows': {
            'name': WINDOWS,
            'username': getpass.getuser()
        },
        'linux': {
            'name': LINUX,
            'username': getpass.getuser()
        },
        'darwin': {
            'name': MAC,
            'username': getpass.getuser()
        }
    }

    _os = platform.system().lower()
    if not _os in OPERATING_SYSTEMS.keys():
        raise UnknownOS("Operating system %s is not currently supported." % platform.system())

    username = OPERATING_SYSTEMS[_os]['username']
    name = OPERATING_SYSTEMS[_os]['name']

    Distribution = get_linux_distribution()

    # for parameter, value in OPERATING_SYSTEMS[_os].iteritems():
    #     locals()[parameter] = value


