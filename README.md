## PROBLEM STATEMENT

In the modern digital landscape, the exponential growth of information has made it increasingly challenging for individuals and organizations to efficiently retrieve, process, and utilize data from various sources. Users often face difficulties in extracting relevant insights from web content, understanding complex documents, and solving programming-related queries etc... This results in wasted time, increased frustration, and an overall decline in productivity. Furthermore, the existing tools for these tasks are often fragmented, requiring users to switch between multiple applications and platforms, which complicates the workflow. The lack of an interactive, user-friendly tool can hinder the ability of users to assess models efficiently, understand their performance, and share insights with stakeholders. Furthermore, generating comprehensive reports that include model evaluation metrics, visualizations, and relevant statistics can be a tedious and time-consuming process, especially for users with limited technical expertise.

## SUGGESTED SOLUTION

To overcome the above problem , we have made a Chatbot i.e ShardAI incorporating various multiple functionalities to perform various tasks that individuals usually face difficulty on. This is a comprehensive application leveraging Langchain and various open source Models(i.e Gemini , Llama etc...) which enhances user productivity and facilitates seamless integration with the Digital content.

To do this , we have equiped our Chatbot with various functionalities like :

### 1. Ask Queries:
This code initializes a chatbot using a language model (LLM) with integrated tools like Wikipedia and Arxiv for retrieving relevant information. It allows users to interact via speech or text, facilitating efficient retrieval and summarization of web content or complex documents. This aligns with the problem statement by addressing the challenge of fragmented tools and improving productivity through a unified, user-friendly platform for information access and report generation.

### 2. PDF Summarisation(with audio functionality) And Ask Questions:
This code enables PDF summarization with audio playback and a question-answering system, allowing users to extract key insights from complex documents and interact with the content via natural language queries. It aligns with the problem statement by streamlining the retrieval and comprehension of information, reducing workflow fragmentation, and enhancing productivity through an integrated, user-friendly tool.

### 3. Career Recommendations System and Generation Of Interview Questions
This code creates a career recommendation system where users upload their resume, and an AI model provides personalized career paths, interview questions, and resume optimization tips. It aligns with the problem statement by simplifying the process of extracting relevant career insights, offering a user-friendly solution to streamline job search and interview preparation, reducing the need for fragmented tools.

### 4. Cold Email Generator with Skill Gap Analysis
This code implements a "Cold Email Generator with Skill Gap Analysis," where users can input a job description URL and compare their skills with the job's requirements. It aligns with the problem statement by generating personalized cold emails that highlight matching and missing skills, helping users apply for jobs effectively while identifying skill gaps.

### 5. ML Model Evaluation and PDF Generation
The code aims to create an interactive tool that simplifies data retrieval, processing, and model evaluation by integrating functionalities like web scraping, document summarization, and programming assistance into a single platform. This directly addresses the problem of fragmented tools and enhances user productivity by providing an intuitive interface for extracting insights and generating comprehensive reports, ultimately streamlining workflows for users with varying technical expertise.

### 6. Code Analyst
The code snippet implements a Streamlit application that allows users to interactively analyze and improve code by sending prompts to a local API. This aligns with the problem statement by providing a user-friendly interface that simplifies the process of retrieving relevant insights from programming-related queries, thereby enhancing productivity and reducing frustration associated with fragmented tools.

### Moving Forward we would be adding more functionalities and integrating Flask Web Framework in place of Streamlit....

## HOW TO RUN THIS PROJECT

1. Clone or Download this Repository to your local machine.
2. Then go to your Command prompt in VS Code and create a virtual environment venv with a python version == 3.12.0 using the command **conda create -p python == 3.12 -y**.
3. Install all the libraries mentioned in the requirements.txt file with the command **pip install -r requirements.txt**.
4. Also install llama2 model on your local machine by writing the command **ollama run llama2**.
5. For using the code-problem solver funnctionality , go to your command promple and go to the loaction of your modelfile using the 'cd' command and then write **ollama create codeguru -f modelfile** and then **ollama run codeguru** this will run the model in background.
6. Open your terminal/command prompt from your project directory and run the file main.py by executing the command **streamlit run main2.py**.
7. Go to your browser and type **http://192.168.174.134:8501**.
8. Hurray! That's it.

## NOTE:
This is just the Prototype of the Model and not the entire code **(i.e we would be incorporating more features)** and also integrating a **FLASK WEB FRAMEWORK** in place of the existing **STREAMLIT FRAMEWORK**.
