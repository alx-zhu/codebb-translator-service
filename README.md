# Translator Service

This repo contains a Python Flask web app that will perform live translations for input text. The repo contains starter code that provides hard-coded dummy translations, which you can modify to include calls to an LLM.

## Fork this repo to use

<img width="200" alt="image" src="https://github.com/CMU-313/translator-service/assets/5557706/47e9c1fb-5b9d-41fc-b825-05994867388a">

# Build and run locally

```
virtualenv .app                    # Create a virtual environment (do this just once in the directory)
source .app/bin/activate           # Start virtual environment (do this every time you use a new terminal tab in this directory)
```

## Installing Dependencies
```
pip install -r requirements.txt    # Do this just once. It will install `flask` and `pytest`
```

## Run tests locally
```
pytest                             # You should see the tests in test_translator.py run and pass successfully
```

## Run the translator service locally

```
flask run                          # Starts a web server on http://127.0.0.1:5000
```

Navigate to [http://127.0.0.1:5000/?content=Dies ist eine Nachricht auf Deutsch](http://127.0.0.1:5000/?content=Dies%20ist%20eine%20Nachricht%20auf%20Deutsch) and you should see the response JSON:

```
{"is_english":false,"translated_content":"This is a German message"}
```

See the code in `src/translator.py` for the full list of hard-coded dummy translations.

# Deploying this service to Azure

Create a new Web App and deploy to Azure. Recall the [steps you used to deploy NodeBB to Azure (section "Create a Web App")](https://docs.google.com/document/d/1cC95F2752ZNmAJ_VPjZmEd8UoUhBi7-lQElx6OaZFd0). The process to deploy the translator service is similar, with a few differences:

1. You do not need a database / Redis cache, just create a new Web App.
2. For the runtime stack, choose "Python 3.10" instead of NodeJS.
3. App Service Plan: Basic B1 should be sufficient
4. On the "Deployment" tab, make sure to enable "Basic Authentication" at the bottom, then enable "Continuous Deployment" at the top, and connect to your fork of this GitHub repository. You can preview the workflow file, which should look similar to [this file](https://github.com/CMU-313/translator-service/blob/f24/.github/workflows/sample-build.yml) but have your own "app-name" and "publish-profile" at the bottom of the file.
5. Once you create the resource, the deployment should ideally work out-of-the-box. Azure can detect that your repo is a Flask app and run it.

## Testing your deployment

Once you have deployed this service, you can access the following link `https://<app-name>.azurewebsites.net/?content=Dies%20ist%20eine%20Nachricht%20auf%20Deutsch` (replace `<app-name>` with your deployed resource name) and you will see a JSON response like the one above.


# Integrating the translator service with NodeBB

Now that you have a dummy translator service deployed, you can integrate it into NodeBB by allowing new posts to be translated at creation time and to display a "Translate" button for such posts. To save you the trouble, we are providing the code changes required for this UI. You need two sets of changes, in the back-end (NodeBB repo) and in the front-end (theme repo):

1. https://github.com/CMU-313/NodeBB/compare/f24...f24-p4
2. https://github.com/CMU-313/nodebb-theme-harmony/compare/f24...f24-p4

You can merge this commit directly if you know how to set up a new remote and perform cherry picking; or you can just look at the diffs above and copy+paste the changes carefully into your own NodeBB repos. These are provided only as suggestions but you are welcome to do something else.

Note that you will have to edit the translator service API URL in your integration: https://github.com/CMU-313/NodeBB/blob/f24-p4/src/translate/index.js#L7

## Testing the integration

Re-deploy your NodeBB after merging all the changes with `./nodebb build` and `./nodebb start` (or `./nodebb dev`). Now, when you create a new post using one of the hard-coded non-English texts they should get translated auotmatically by the back-end:

![image](https://github.com/user-attachments/assets/61f1d9ca-3ca4-4a68-8869-d381d3d06ac6)

After submitting...

![image](https://github.com/user-attachments/assets/f07d51ea-217a-44d8-a314-62bbe1a4cee4)

Clicking the button reveals...

![image](https://github.com/user-attachments/assets/1e804235-684f-46fd-b016-0d3dd3297991)


# Implementing the LLM based translator

Please replace `translate` method in `src/translator.py` with your LLM based
implementation. The `translate` method takes a string `content` as input and
returns a tuple `(bool, str)`, indicating if `content` is in English and
the translated content if `content` is not in English.


> [!WARNING]
> Do not push your API key to your repository. You should use environment variables to store your API key.

## Handle responses from the LLM

You need to design your prompt so that you can parse the result from an LLM model.
However, your system needs to be robust enough to recover if the LLM does not respond as you expect.
It is up to you how your system reacts to unexpected responses. You can try a different prompt, return an error message, or simply assume the input is in English.

# Testing your implementation

Now you need to test your implementation.

To do this, please complete the unit test in `test/unit/test_translator.py`.
In `test_llm_normal_response()`, please implement a unit test that verifies that
your program can return correct value if LLM provides an expected result.
In `test_llm_gibberish_response()`, please implement a unit test that verifies
that your program can handle a gibberish response.
