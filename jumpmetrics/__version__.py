"""Module providing a function specifying the package version."""
__version__ = '0.1.0'
SHORT_VERSION = __version__


def parse_version_info(version_str: str) -> tuple:
    """Function to parse the version info

    Args:
        version_str (str): version string of the current package version

    Returns:
        tuple: Version information of the current package
    """
    ver_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            ver_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            ver_info.append(int(patch_version[0]))
            ver_info.append(f'rc{patch_version[1]}')
    return tuple(ver_info)


version_info = parse_version_info(__version__)
