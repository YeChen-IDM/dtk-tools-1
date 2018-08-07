import os
import platform
import getpass
from simtools.Utilities import Distro


def get_linux_distribution():
    name = Distro.name().lower()

    if 'centos' in name:
        return 'CentOS'
    elif 'ubuntu' in name:
        return 'Ubuntu'
    elif 'debian' in name:
        return 'Debian'
    elif 'fedora' in name:
        return 'Fedora'


def command_exist(program):
    """
    Finds if a program exists in the path on the system
    :param program: The program we want to find
    :return: True if the program exists, False if not
    """
    def is_exe(fpath):
        """
        Tests if the file exists and is an executable
        """
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    # For each directory in the path, check if the program exists there
    for path in os.environ["PATH"].split(os.pathsep):
        exe_file = os.path.join(path, program)
        if is_exe(exe_file):
            return True


class LocalOS:
    """
    A Central class for representing values whose proper access methods may differ between platforms.
    """
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
    PIP_COMMANDS = ['pip3.7', 'pip3.6', 'pip3', 'pip']

    _os = platform.system().lower()
    if not _os in OPERATING_SYSTEMS.keys():
        raise UnknownOS("Operating system %s is not currently supported." % platform.system())

    username = OPERATING_SYSTEMS[_os]['username']
    name = OPERATING_SYSTEMS[_os]['name']

    Distribution = get_linux_distribution()

    @classmethod
    def get_pip_command(cls):
        # If we are on windows, use .exe at the end of the command
        if cls.name == cls.WINDOWS:
            commands = ["{}.exe".format(c) for c in cls.PIP_COMMANDS]
        else:
            commands = cls.PIP_COMMANDS

        for pip in commands:
            if command_exist(pip):
                return pip

        # If we get to this point, no pip was found -> exception
        raise OSError("pip could not be found on this system.\n"
                      "Make sure Python is installed correctly and pip is in the PATH")




