import os
import yaml
import json
import openai
from flask import jsonify, request
import pinecone
from dotenv import load_dotenv
from .init_flask_app import app
from .local_database.database_models import Conversation, User, Document
from .utils.agent_utils import (
    extract_openai_chat_messages,
    cleanup_function_call_messages,
)

# from .agents.openai_agents.openai_functions_agent import OpenAIFunctionsAgent
from .agents.agent_factory import AgentFactory
from .data_structures.data_structures import (
    PineconeConfig,
    ChromaDBConfig,
    DataProcessingConfig,
    DocumentProcessingConfig,
    ConfigFileValidator,
)
from .vector_database.pinecone.pinecone_database_handler import PineconeDatabaseHandler
from .vector_database.chroma_db.chroma_db_database_handler import ChromaDatabaseHandler
from .vector_database.pinecone.generate_pinecone_db import (
    generate_database,
    update_database,
)
from .utils.data_processing_utils import (
    extract_meta_data,
    remove_meta_data_from_text,
    split_text_into_parts_and_chapters,
)
from .llm_functionalities.openai_task_functions import summarize_text


@app.route("/", methods=["GET"])
def home():
    print("Hello world")
    return "Hello world"


@app.route("/register_new_user", methods=["POST"])
def register_new_user():
    """Checks if username and password are correct"""
    request_data = json.loads(request.data)
    if (
        not request_data
        or "username" not in request_data
        or "password" not in request_data
    ):
        return (
            jsonify({"error": "Request must contain a 'username' and a 'password'"}),
            400,
        )
    username = request_data["username"]
    password = request_data["password"]
    user_exists = User.check_user_exists(username)
    if not user_exists:
        user_id = User.save_new_user(username, password)
        return jsonify({"user_id": user_id})
    else:
        return jsonify({"error": "User already exists"})


@app.route("/check_username_exists", methods=["POST"])
def check_username_exists():
    """Checks if the username already exists"""
    request_data = json.loads(request.data)
    if not request_data or "username" not in request_data:
        return (
            jsonify({"error": "Request must contain a 'username' and a 'password'"}),
            400,
        )
    username = request_data["username"]
    user_exists = User.check_user_exists(username)
    return jsonify({"user_exists": user_exists})


@app.route("/check_username_and_password", methods=["POST"])
def check_username_and_password():
    """Checks if username and password are correct"""
    request_data = json.loads(request.data)
    if (
        not request_data
        or "username" not in request_data
        or "password" not in request_data
    ):
        return (
            jsonify({"error": "Request must contain a 'username' and a 'password'"}),
            400,
        )
    username = request_data["username"]
    password = request_data["password"]
    user_id = User.check_username_and_password(username, password)
    return jsonify({"user_id": user_id})


@app.route("/execute_rag", methods=["POST"])
def execute_rag():
    """
    Handles the conversation with an AI agent. It processes the incoming query,
    retrieves or starts a new conversation, generates a response using the AI model,
    and returns the AI's response along with context information.

    Returns:
        JSON response containing the answer from the AI agent and any source information used.
    """
    request_data = json.loads(request.data)
    if not request_data or "query" not in request_data or "user_id" not in request_data:
        return jsonify({"error": "Request must contain a 'query' key"}), 400

    query = request_data["query"]
    user_id = request_data["user_id"]
    conv_id = Conversation.get_latest_conversation(user_id=user_id)
    if conv_id == None:
        conv_id = Conversation.generate_new_conversation(user_id=user_id)
        if not conv_id:
            return (
                jsonify(
                    {"error": "An error occured when generating a new conversation"}
                ),
                400,
            )
    chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
    rag_model = AgentFactory.create_agent()
    chat_messages = extract_openai_chat_messages(chat_messages=chat_messages)
    agent_answer, chat_messages = rag_model.run(
        query=query, chat_messages=chat_messages
    )
    meta_data = rag_model.get_meta_data()
    chat_messages = cleanup_function_call_messages(chat_messages=chat_messages)
    chat_messages.append(
        {
            "role": "assistant",
            "content": agent_answer.final_answer,
        }
    )
    Conversation.update_chat_messages(conv_id=conv_id, chat_messages=chat_messages)
    return jsonify({"answer": agent_answer.final_answer, "meta_data": meta_data})


@app.route("/test_summarization", methods=["POST"])
def test_summarization():
    request_data = json.loads(request.data)
    text = """Jedem Autor, vermute ich mal, schwebt eine Situation vor, in der Leser seines Werks von der Lektüre desselben profitieren könnten. Ich denke dabei an den Kaffeeautomaten im Büro, vor dem Mitarbeiter Ansichten und Tratsch mitei- nander austauschen. Meine Hoffnung ist, dass ich den Wortschatz bereichere, den Menschen benutzen, wenn sie sich über Urteile und Entscheidungen ande- rer, die neue Geschäftsstrategie ihres Unternehmens oder die Anlageentschei- dungen eines Kollegen unterhalten. Weshalb sich mit Tratsch befassen? Weil es viel leichter und auch viel angenehmer ist, die Fehler anderer zu erkennen und zu benennen als seine eigenen. Selbst unter den günstigsten Umständen fållt es uns schwer, unsere Uberzeugungen und VWünsche zu hinterfragen, und es fällt uns besonders schwer, wenn es am nötigsten wäre - aber wir können von den sachlich fundierten Meinungen anderer profitieren. Viele von uns nehmen in Gedanken von sich aus vorweg, wie Freunde und Kolegen unsere Entscheidun- gen beurteilen werden; deshalb kommt es maßgeblich auf Qualität und Inhalt dieser vorweggenommenen Urteile an. Die Erwartung intelligenten Geredes über uns ist ein starkes Motiv für ernsthafte Selbstkritik, stärker als alle an Silvester gefassten guten Vorsätze, die Entscheidungsfindung am Arbeitsplatz und zu Hause zu verbessern.
Um zuverlässige Diagnosen zu stellen, muss ein Arzt eine Vielzahl von Krankheitsbezeichnungen lernen, und jeder dieser Termini verknüpft ein Kon- zept der Erkrankung mit ihren Symptomen, möglichen Vorstufen und Ursa- chen, möglichen Verläufen und Konsequenzen sowie möglichen Eingriffen zur Heilung oder Linderung der Krankheit. Das Erlernen der ärztlichen Heilkunst besteht auch darin, die medizinische Fachsprache zu erlernen. Um Urteile und Entscheidungen besser verstehen zu können, bedarf es eines reichhaltigeren Wortschatzes, als ihn die Alltagssprache zur Verfügung stellt. Die Tatsache, dass unsere Fehler charakteristische Muster aufweisen, begründet die Hoffnung darauf, dass andere in sachlich fundierter Weise über uns reden mögen. Syste- matische Fehler - auch »Verzerrungen« (biases) genannt - treten in vorherseh- barer Weise unter bestimmten Umnständen auf. Wenn ein attraktiver und selbstbewusster Redner dynamisch aufs Podium springt, kann man davon aus- gehen, das das Publikum seine Äußerungen günstiger beurteilt, als er es eigent- lich verdient. Die Verfügbarkeit eines diagnostischen Etiketts für diesen syste- natischen Fehler- der Halo-Efekt - erleichtert es, ihn vorwegzunehmen erkennen und zu verstehen. Wenn Sie gefragt werden, woran Sie gerade denken, können Sie diese Frage normalerweise beantworten. Sie glauben zu wissen, was in Ihrem Kopf vor sich geht - oftmals führt ein bewusster Gedanke in wohlgeordneter Weise zum nächsten. Aber das ist nicht die einzige Art und Weise, wie unser Denkver- mögen (mind) funktioniert, es ist nicht einmal seine typische Funktionsweise. Die meisten Eindrücke und Gedanken tauchen in unserem Bewusstsein auf, ohne dass wir wüssten, wie sie dorthin gelangten. Sie können nicht rekonstru- ieren, wie Sie zu der Überzeugung gelangten, eine Lampe stehe auf dem Schreibtisch vor Ihnen, wie es kam, dass Sie eine Spur von Verärgerung aus der Stimme Ihres Gatten am Telefon heraushörten, oder wie es Ihnen gelang, einer Gefahr auf der Straße auszuweichen, ehe Sie sich ihrer bewusst wurden. Die mentale Arbeit, die Eindrücke, Intuitionen und viele Entscheidungen hervor- bringt, vollzieht sich im Stillen in unserem Geist. Ein Schwerpunkt dieses Buches sind Fehler in unserem intuitiven Denken. Doch die Konzentration auf diese Fehler bedeutet keine Herabsetzung der menschlichen Intelligenz, ebenso wenig, wie das Interesse an Krankheiten in medizinischen Texten Gesundheit verleugnet. Die meisten von uns sind die meiste Zeit ihres Lebens gesund, und die meisten unserer Urteile und Handlun- gen sind meistens angemessen. Auf unserem Weg durchs Leben lassen wir uns normalerweise von Eindrücken und Gefühlen leiten, und das Vertrauen, das wir in unsere intuitiven Uberzeugungen und Präferenzen setzen, ist in der Regel gerechtfertigt. Aber nicht immer. Wir sind oft selbst dann von ihrer Richtigkeit überzeugt, wenn wir irren, und ein objektiver Beobachter erkennt unsere Fehler mit höherer Wahrscheinlichkeit als wir selbst. Und so wünsche ich mir, dass dieses Buch die Gespräche am Kaffeeauto- maten dadurch verändert, dass es unsere Fähigkeit verbessert, Urteils- und Ent- scheidungsfehler von anderen und schließlich auch von uns selbst zu erkennen und verstehen, indem es dem Leser eine differenzierte und exakte Sprache an die Hand gibt, in der sich diese Fehler diskutieren lassen. Eine zutreffende Diagnose mag wenigstens in einigen Fällen eine Korrektur ermöglichen, um den Schaden, den Fehlurteile und -entscheidungen verursachen, zu begrenzen.
Dieses Buch stellt mein gegenwärtiges Verständnis von Urteils- und Entschei- dungsprozessen dar, das maßgeblich von psychologischen Entdeckungen der letzten Jabrzehnte geprägt wurde. Die zentralen Ideen gehen allerdings auf jenen glücklichen Tag des Jahres 1969 zurück, an dem ich einen Kollegen bat, als Gastredner in einem Seminar zu sprechen, das ich am Fachbereich Psycho- logie der Hebräischen Universität von Jerusalem hielt. Amos Tversky galt als ein aufstrebender Star auf dem Gebiet der Entscheidungsforschung - ja, auf allen Forschungsfeldern, auf denen er sich tummelte -, sodass ich wusste, dass es eine interessante Veranstaltung werden würde. Viele Menschen, die Amos kannten, hielten ihn für die intelligenteste Person, der sie je begegnet waren. Er war brillant, redegewandt und charismatisch. Er war auch mit einem voll- kommenen Gedächtnis für Witze gesegnet und mit einer außergewöhnlichen Fähigkeit, mit ihrer Hilfe ein Argument zu verdeurlichen. In Amos Gegenwart war es nie langweilig. Er war damals 32, ich war 35. Amos berichtete den Se- minarteilnehmern von einem aktuellen Forschungsprogramm an der Univer- sität Michigan, bei dem es um die Beantwortung der folgenden Frage ging: Sind Menschen gute intuitive Statistiker? Wir wussten bereits, dass Menschen gute intuitive Grammatiker sind: Ein vierjähriges Kind befolgt, wenn es spricht, mühelos die Regeln der Grammatik, obwohl es die Regeln als solche nicht kennt. Haben Menschen ein ähnlich intuitives Gespür für die grund- legenden Prinzipien der Statistik? Amos berichtete, die Antwort darauf sei ein bedingtes Ja. Wir hatten im Seminar eine lebhafte Diskussion, und wir ver- ständigten uns schließlich darauf, dass ein bedingres Nein eine bessere Ant- wort sei. Amos und mir machte dieser Meinungsaustausch großen Spaß, und wir gelangten zu dem Schluss, dass intuitive Statistik ein interessantes For- schungsgebiet sei und dass es uns reizen würde, dieses Feld gemeinsam zu er- forschen. An jenem Freitag trafen wir uns zum Mittagessen im Café Rimon, dem Stammlokal von Künstlern und Professoren in Jerusalem, und planten eine Studie über die statistischen Intuitionen von Wissenschaftlern. Wir wa- ren in diesem Seminar zu dem Schluss gelangt, dass unsere eigene Intuition unzureichend war. Obwohl wir beide schon jahrelang Statistik lehrten und anwandren, hatten wir kein intuitives Gespür für die Zuverlässigkeit statisti- scher Ergebnisse bei kleinen Stichproben entwickelt. Unsere subjektiven Ur- teile waren verzerrt: Wir schenkten allzu bereitwillig Forschungsergebnissen Glauben, die auf unzureichender Datengrundlage basierten, und neigten dazu, bei unseren eigenen Forschungsarbeiten zu wenig Beobachtungsdaten zu er- heben.! Mit unserer Studie wollten wir herausfinden, ob andere Forscher an der gleichen Schwäche litten. Wir bereiteten eine Umfrage vor, die realistische Szenarien statistischer Probleme beinhaltete, die in der Forschung auftreten. Amos trug die Ant- worten einer Gruppe von Experten zusammen, die an einer Tagung der So- ciety of Mathematical Psychology teilnahmen, darunter waren auch die Ver- fasser zweier Statistik-Lehrbücher. Wie erwartet fanden wir heraus, dass unsere Fachkollegen, genauso wie wir, die Wahrscheinlichkeit, dass das ur- sprüngliche Ergebnis eines Experiments auch bei einer kleinen Stichprobe erfolgreich reproduziert werden würde, enorm überschāzten. Auch gaben sie einer fiktiven Studentin sehr ungenaue Auskünfte über die Anzahl der Beob- achtungsdaten, die sie erheben müsse, um zu einer gültigen Schlussfolgerung zu gelangen. Selbst Statistiker waren also keine guten intuitiven Statistiker. Als wir den Artikel schrieben, in dem wir diese Ergebnisse darlegten, stellten Amos und ich fest, dass uns die Zusammenarbeit großen Spaß machte. Amos war immer sehr witzig, und in seiner Gegenwart wurde auch ich witzig, sodass wir Stunden gewissenhafter Arbeit in fortwährender Erheiterung verbrachten. Die Freude, die wir aus unserer Zusammenarbeit zogen, machte uns unge- wöhnlich geduldig; man strebt viel eher nach Perfektion, wenn man sich nicht langweilt. Am wichtigsten war vielleicht, dass wir unsere kritischen Waffen an der Tür abgaben. Sowohl Amos als auch ich waren kritisch und streitlustig - er noch mehr als ich, aber in den Jahren unserer Zusammenarbeit hat keiner von uns beiden irgendetwas, was der andere sagte, rundweg abgelehnt. Eine der größten Freuden, die mir die Zusammenarbeit mit Amos schenkte, bestand gerade darin, dass er viel deutlicher als ich selbst sah, worauf ich mit meinen vagen Gedanken hinauswollte. Amos war der bessere Logiker von uns beiden, er war theoretisch versierter und hatte einen untrüglichen Orientierungssinn. Ich hatte einen intuitiveren Zugang und war stärker in der Wahrnehmungs- psychologie verwurzelt, aus der wir viele Ideen übernahmen. Wir waren ein- ander hinreichend ähnlich, um uns mühelos zu verständigen, und wir waren hinreichend unterschiedlich, um uns gegenseitig zu überraschen. Wir ver- brachten routinemäßig einen Großteil unserer Arbeitstage zusammen, oftmals auf langen Spaziergängen. Während der kommenden 14 Jahre bildete diese Zusammenarbeit den Mittelpunkt unserer Leben, und unsere gemeinsamen Arbeiten aus dieser Zeit waren das Beste, was jeder von uns überhaupt an wissenschaftlichen Beiträgen lieferte.
Wir entwickelten schon bald eine bestimmte Vorgehensweise, die wir viele Jahre lang beibehielten. Unsere Forschung bestand in einem Gespräch, in dem wir Fragen erfanden und gemeinsam unsere intuitiven Antworten überprüften. Jede Frage war ein kleines Experiment, und wir führten an jedem Tag viele Experimente durch. Wir suchten nicht ernsthaft nach der richtigen Antwort auf die statistischen Fragen, die wir stellten. Wir wollten die intuitive Antwort herausfinden und analysieren, die erste, die uns einfiel, diejenige, die wir spon- tan geben wollten, auch wenn wir wussten, dass sie falsch war. Wir glaubten- richtigerweise, wie sich zeigte -, dass jede Intuition, die wir beide teilten, auch von vielen anderen geteilt würde und dass es leicht wäre, ihre Auswirkungen auf Urteile nachzuweisen. Einmal entdeckten wir zu unserer großen Freude, dass wir die gleichen verrückten Ideen über die zukünftigen Berufe mehrerer Klein- kinder hatten, die wir beide kannten. Wir identifizierten den streitlustigen drei- jährigen Anwalt, den schrulligen Professor, den empathischen und leicht zu- dringlichen Psychotherapeuten. Natürlich waren diese Vorhersagen absurd, aber wir fanden sie trotzdem reizvoll. Es war auch klar, dass unsere Intuitionen von der Ähnlichkeit beeinflusst wurden, die das jeweilige Kind mit dem kultu- rellen Stereotyp eines bestimmten Berufs aufwies. Diese lustige Übung half uns dabei, eine Theorie zu entwickeln, die damals in unseren Köpfen im Entstehen begriffen war, und zwar über die Bedeutung der Ähnlichkeit bei Vorhersagen. Wir überprüften und verfeinerten diese Theorie in Dutzenden von Experimen- ten, wie etwa dem folgenden.
	Wenn Sie über die nächste Frage nachdenken, sollten Sie davon ausgehen, dass Steve zufällig aus einer repräsentativen Stichprobe ausgewählt wurde:

	Eine Person wurde von einem Nachbarn wie folgt beschrieben: » Steve ist sehr scheu und verschlossen, immer hilfsbereit, aber kaum an anderen oder an der Wirklichkeit interessiert. Als sanftmütiger und ordentlicher Mensch hat er ein Bedürfnis nach Ordnung und Struktur und eine Pas- sion für Details.« Ist Steve eher Bibliothekar oder eher Landwirt?"""
    response = summarize_text(text=text)
    return response


@app.route("/get_latest_conv_id", methods=["POST"])
def get_latest_conv_id():
    """
    Returns:
    The latest conversation id
    """
    request_data = json.loads(request.data)
    if not request_data or "user_id" not in request_data:
        return jsonify({"error": "Request must contain a 'user_id' key"}), 400
    user_id = request_data["user_id"]
    return jsonify({"conv_id": Conversation.get_latest_conversation(user_id=user_id)})


@app.route("/create_new_conversation", methods=["POST"])
def create_new_conversation():
    """
    Creates a new conversation for a user and returns the conversation ID.

    Returns:
        String indicating the creation of a new conversation with its ID.
    """
    request_data = json.loads(request.data)
    if not request_data or "user_id" not in request_data:
        return jsonify({"error": "Request must contain a 'user_id' key"}), 400
    user_id = request_data["user_id"]
    conv_id = Conversation.generate_new_conversation(user_id=user_id)
    return jsonify({"conv_id": conv_id})


@app.route("/get_chat_messages", methods=["POST"])
def get_chat_messages():
    """
    Retrieves chat messages for a given conversation ID.

    Returns:
        JSON response containing all chat messages of the requested conversation.
        In case of an error, returns an error message.
    """
    request_data = json.loads(request.data)
    if not "query" in request_data:
        return "Request must contain a 'query' key"
    conv_id = request_data["query"]
    try:
        chat_messages = Conversation.get_chat_messages(conv_id=conv_id)
        return jsonify(chat_messages)
    except Exception as e:
        print(e)
        return f"An error occured {e}"


@app.route("/generate_vector_db", methods=["GET"])
def generate_vector_db():
    """
    Generate the vector database.

    Returns:
        JSON response containing the information of the created database.
    """
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    # pinecone_config = PineconeConfig(**config_data["pinecone_db"])
    chroma_db_config = ChromaDBConfig(**config_data["chroma_db"])
    # database_handler = PineconeDatabaseHandler(
    #     index=pinecone.Index(pinecone_config.index_name),
    #     data_processing_config=data_processing_config,
    #     pinecone_config=PineconeConfig(**config_data["pinecone_db"]),
    # )
    database_handler = ChromaDatabaseHandler(
        chroma_db_config=chroma_db_config, data_processing_config=data_processing_config
    )
    generate_database(database_handler=database_handler)
    return f"Database generated"


@app.route("/upload_document", methods=["POST"])
def upload_document():
    with open(os.getenv("CONFIG_FP"), "r") as file:
        config_data = yaml.safe_load(file)
    data_processing_config = DataProcessingConfig(**config_data["data_processing"])
    pinecone_config = PineconeConfig(**config_data["pinecone_db"])
    database_handler = PineconeDatabaseHandler(
        index=pinecone.Index(pinecone_config.index_name),
        data_processing_config=data_processing_config,
        pinecone_config=pinecone_config,
    )
    uploaded_text = request.data.decode("utf-8")
    document_config = DocumentProcessingConfig(**config_data["document_processing"])
    meta_data = extract_meta_data(
        extraction_pattern=document_config.meta_data_pattern,
        document_text=uploaded_text,
    )
    uploaded_text = remove_meta_data_from_text(text=uploaded_text)
    update_database(
        text=uploaded_text,
        text_meta_data=meta_data,
        database_handler=database_handler,
        document_processing_config=document_config,
    )
    document_id = Document.save_document(
        user_id=meta_data["user_id"],
        document_type=meta_data["type"],
        document_text=uploaded_text,
    )
    return f"Inserted document id {document_id}"


def main():
    """
    Main function to initialize the Flask application.
    It sets up environment variables, loads configuration, and starts the Flask app.
    """
    load_dotenv()
    with open(
        os.environ["CONFIG_FP"],
        "r",
    ) as file:
        config_data = yaml.safe_load(file)
    try:
        ConfigFileValidator(
            usage_settings=config_data["usage_settings"],
            data_processing_config=DataProcessingConfig(
                **config_data["data_processing"]
            ),
            document_processing_config=DocumentProcessingConfig(
                **config_data["document_processing"]
            ),
            chroma_db_config=ChromaDBConfig(**config_data["chroma_db"]),
            pinecone_db_config=PineconeConfig(**config_data["pinecone_db"]),
            prompt_configs_fp=os.getenv("PROMPT_CONFIGS_FP"),
        )
    except AssertionError as e:
        print(e)
        print("Error in config file validation occured!")
        return
    except ValueError as e:
        print(e)
        print("Error in config file validation occured!")
        return

    if config_data["usage_settings"]["llm_service"] == "openai":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
        )
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
