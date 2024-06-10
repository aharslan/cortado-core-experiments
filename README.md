# Cortado-Core-Experiments

This repository contains the experiments related to [cortado-core](https://github.com/cortado-tool/cortado-core).

## Setup
* Install Python 3.10.x (https://www.python.org/downloads/). Make sure to install a 64-BIT version.
* Optional (recommended): Create a virtual environment (https://docs.python.org/3/library/venv.html) and activate it
* Install all packages required by the experiments
  * Execute `pip install -r requirements.txt`

## Contributing

### Code Quality

We highly value code quality and reliability in our project. To ensure this, our GitLab pipeline includes linting using `black`.

### GitLab Pipeline

Our GitLab pipeline automatically performs essential checks whenever code changes are pushed to the repository.

#### Code Linting

We're committed to maintaining consistent code formatting and style. The pipeline includes a linting stage that uses `black`, a powerful code formatter for Python. This ensures that our code adheres to a unified and clean style, enhancing readability and maintainability.

### Development and Code Formatting
During development, we encourage you to utilize black for code formatting. The black tool helps maintain a consistent style across our codebase and minimizes formatting-related discussions. It's recommended to run black on your code before committing changes. You can do so using the following command:
`black .`

By incorporating black into your workflow, you contribute to maintaining a clean and organized codebase.