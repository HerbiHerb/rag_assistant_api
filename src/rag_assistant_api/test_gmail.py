from .agents.langchain.langchain_tools.tools import GetNewEmails, GoogleSearch


if __name__ == "__main__":
    # email_func = GetNewEmails()

    # email_func._run()

    # test = 0
    google_search = GoogleSearch()
    google_search._run(search_term="Was hat Elon Musk studiert?")
