from rich import print


def cprint(message, color='white'):
    """Simple colored print function"""
    print(f'[{color}]{message}[/{color}]')
