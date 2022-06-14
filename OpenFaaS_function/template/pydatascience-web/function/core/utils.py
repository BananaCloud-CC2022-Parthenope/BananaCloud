

def getSecret(secret_name: str) -> str:
    """load secret value from the openfaas secret folder
    Args:
        secret_name (str): name of the secret
    """
    with open("/var/openfaas/secrets/" + secret_name, "r") as file:
        return file.read()
