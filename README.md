<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h1 align="center">Rag-Assistant-API</h1>

  <!-- <p align="center">
    An interface to calculate an individual price for a customer
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p> -->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!-- <li><a href="#structure">Structure</a></li> -->
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <!-- <li><a href="#license">License</a></li> -->
    <!-- <li><a href="#contact">Contact</a></li> -->
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

The RAG-Assistant-API is a project which makes it easy to generate and use new agents for a wide variety of tasks. Currently it is possible to use LLMs  either directly from OpenAI or from Azure OpenAI but it could be exended to other plathforms as well. For each agent it is possible to implement new functions he can use to answer the incoming user query. It is possible to use the azure assistants api as well. The only thing you need to do is to define an assistant in the azure ai studio,  paste the assistant_id into the config.yaml file and define the AzureOpenAIAssistant as the agent_type.

<!-- ![image info](readme_images/cbp_overview.PNG) -->


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
<!-- * [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url] -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To implement new functionalities into the api you first need a python environment with the neccessary packages. With poetry it is very easy to get all the neccessary packages in the right version installed on your machine. But if you dont have poetry and python installed, follow the steps below.

### Prerequisites

*   <strong>Install Poetry in Linux</strong>

    Open a shell and execute the following command ([Poetry-Installation](https://www.digitalocean.com/community/tutorials/how-to-install-poetry-to-manage-python-dependencies-on-ubuntu-22-04)):
    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Add the following line to the ~/.bashrc file:
    ```
    export PATH="/home/"user_id"/.local/bin:$PATH" (replace "user_id" with your user id)
    ```

    Reload the bashrc file: 
    ```
    source ~/.bashrc 
    ```

    Test the poetry installation with 
    ```
    poetry --version
    ```

*   <strong>Install PyEnv</strong>

    PyEnv is a tool with which you can easily install new python versions if needed.

    To install PyEnv execute the following commands:
    ```
    sudo apt update
    ```
    ```
    sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```
    ```
    curl https://pyenv.run | bash
    ```

    Add the following lines to the ~/.bashrc file:
    ```
    export PYENV_ROOT="$HOME/.pyenv"
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    ```

    <strong>Generate a .env file in the root folder of the project</strong>

    The .env file contains all necessary environment variables. The file should contain the following variables

    Variables to define the config file paths:
    CONFIG_FP="config/config.yaml"
    PROMPT_CONFIGS_FP = "config/prompt_configs.yaml"

    Variables for the openai or azure openai keys:
    OPENAI_API_KEY="your api key"
    OPENAI_API_TYPE = azure or openai
    OPENAI_API_VERSION = 2024-05-01-preview
    AZURE_OPENAI_ENDPOINT = "The azure openai endpoint"

    PINECONE_API_KEY="your pinecone api key"
    PINECONE_ENVIRONMENT="gcp-starter"


    <strong>Start the API</strong>

    Set path to the python environment inside the IDE and start the main.py file. Then the flask application should start. 

*   <strong>Install docker-compose</strong>

    Based on [Docker-Compose](https://docs.docker.com/compose/install/standalone/) execute the following command:
    ```
    curl -SL https://github.com/docker/compose/releases/download/v2.28.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
    ```

    Make sure you have docker also installed!

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:HerbiHerb/rag_assistant_api.git
   ```
2. Install python3.11 with pyenv
   ```sh
   pyenv install 3.11
   ```
   Make the installed python version global
   ```sh
   pyenv global 3.11
   ```
3. Navigate into the cloned repository cbp_api
4. Install all the necessary packages with poetry
    ```sh
    poetry install
    ```
    Activate the poetry environment with
    ```sh
    poetry shell
    ```
    Get the path to the installed environment and copy it
    ```sh
    poetry show -v
    ```
    Add the path to the python interpreter path in e.g. VSCode (Strg + Shift + P -> select interpreter -> enter interpreter path)
5.  Create the docker image
    ```sh
    docker build --no-cache -t rag-assistant .
    ```
6. Run the docker containers with docker-compose or docker
   ```sh
   docker-compose up --build -d
   ```
   ```sh
   docker run -p 5000:5000 rag-assistant
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

If the application is running you can access the endpoints with curls commands

1.  Register a new user:
    The interact with the application a user is required and to generate a new user you can call the 'register_new_user' endpoint
    ```sh
    curl --noproxy "*" -d '{"username":"NewUser", "password": "1234"}' -H "Content-Type: application/json" -X POST http://localhost:5000/register_new_user
    ```
    The API returns the new user_id you can use in other endpoints.
2.  Get the latest conversation id from a user: 
    ```sh
    curl --noproxy "*" -d '{"user_id": "1"}' -H "Content-Type: application/json" -X POST http://localhost:5000/get_latest_conv_id
    ```
3.  Create a new conversation for a user: 
    ```sh
    curl --noproxy "*" -d '{"user_id": "1"}' -H "Content-Type: application/json" -X POST http://localhost:5000/create_new_conversation
    ```
4.  Get the chat messages to a conversation: 
    ```sh
    curl --noproxy "*" -d '{"query": "1"}' -H "Content-Type: application/json" -X POST http://localhost:5000/get_chat_messages
    ```
5.  Generate the vector database: 
    To trigger the generation of the vector database you can call the following endpoint. The corresponding files for the vector database should be stored in the data folder. And the path to the files should be declared in the config.yaml file.
    ```sh
    curl --noproxy "*" -H "Content-Type: application/json" -X GET http://localhost:5000/generate_vector_db
    ```
6.  Chat with the assistant:
    To chat with the assistant you can enter the following curl command
    ```sh
    curl --noproxy "*" -d '{"query":"<any query to the assistant", "user_id":1}' -H "Content-Type: application/json" -X POST http://localhost:5000/execute_rag
    ```

<!-- ROADMAP -->
## Roadmap

- [ ] Doing refactoring steps
- [ ] Adding a basic UI as a chat interface
- [ ] Adding support for other cloud-services like aws, gcp


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
<!-- ## Contact

Dennis Herbrik - dennisherbrik1988@gmail.com -->

<!-- Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name) -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 


