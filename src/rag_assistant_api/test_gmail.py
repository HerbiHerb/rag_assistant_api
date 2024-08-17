from .agents.langchain.langchain_tools.tools import GetNewEmails


if __name__ == "__main__":
    email_func = GetNewEmails()

    email_func._run()

    test = 0
