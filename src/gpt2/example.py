"""A few example methods to get CI/CD setup"""

def echo(msg: str):
    """Echo a message to the console"""
    return msg

def check_prime(num: int) -> bool:
    """Check if a number is prime"""
    if num <= 1:
        return False
    return all(num % i != 0 for i in range(2, int(num ** 0.5) + 1))
