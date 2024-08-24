from .agents.langchain.langchain_tools.tools import (
    GetNewEmails,
    GoogleSearch,
    SendEmail,
    YouTubeSearch,
)


if __name__ == "__main__":
    # email_func = GetNewEmails()

    # email_func._run()

    # send_mail = SendEmail()

    # send_mail._run(subject="Erinnerung", message="Rufe das Krankenhaus an")

    test = 0
    google_search = GoogleSearch()
    google_search._run(search_term="Was hat Elon Musk studiert?")
