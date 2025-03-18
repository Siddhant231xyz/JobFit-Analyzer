<details>
<summary>LICENSE</summary>

MIT License

Copyright (c) 2024 JobFit Analyzer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

</details>

# JobFit Analyzer 🎯

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

## Overview 📋
![image](https://github.com/user-attachments/assets/44e7bee2-77fd-46ed-b1de-35a17dc61acf)

JobFit Analyzer is an AI-powered web application that revolutionizes the recruitment process by automatically analyzing job descriptions and candidate resumes. Built with Streamlit and leveraging advanced embedding techniques, it provides instant similarity scores and interactive Q&A capabilities to help recruiters make informed decisions.

![JobFit Analyzer Demo](assets/demo.gif)

## Features ✨

- 📄 **Smart Document Processing**
  - Upload and analyze Job Description PDFs
  - Process multiple candidate resumes
  - Automatic embedding computation

- 🔍 **Advanced Analysis**
  - Real-time similarity scoring
  - Multiple query modes for comprehensive analysis
  - RAG-prompt integration for intelligent responses

- ⚡ **Performance Optimization**
  - Efficient session management
  - Single-pass processing
  - Optimized data reuse

- 🚀 **Deployment Options**
  - Local development setup
  - Docker containerization
  - AWS cloud deployment with S3 integration

## Quick Start 🚀

### Prerequisites

- Python 3.9+
- Git
- OpenAI API key

### Local Installation

1. Clone the repository:

```bash
git clone https://github.com/Siddhant231xyz/jobfit-analyzer.git
cd jobfit-analyzer
```

2. Set up virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Local Usage

1. Start the application:

```bash
streamlit run app_local.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Deployment Options 🌐

### Docker Deployment

1. **Build the Docker Image**:
```bash
docker build -t streamlit-resume-app .
```

2. **Create a Docker Volume** for resumes:
```bash
docker volume create resume-volume
```

3. **Run the Container** with the volume mounted:
```bash
docker run -d -p 8501:8501 -v resume-volume:/data/resume streamlit-resume-app
```

### AWS Deployment

#### EC2 Setup

1. **Launch an EC2 Instance** with Amazon Linux 2 AMI
2. **Install Docker** on the EC2 instance:
```bash
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user
```

3. **Install AWS CLI**:
```bash
sudo yum install aws-cli -y
```

4. **Verify SSM Agent is running**:
```bash
sudo systemctl status amazon-ssm-agent
```

#### S3 Integration

1. **Create an S3 bucket** for storing resumes (e.g., `resume-storage-bucket-1`)
2. **Configure S3 Event Notifications** to trigger Lambda function on file uploads
3. **Use Lambda Function** (provided in `Lambda_function.py`) to sync S3 content to Docker volume

#### IAM Setup

- **EC2 Instance Role**: Attach role with `AmazonSSMManagedInstanceCore` policy
- **Lambda Function Role**: Create role with permissions to invoke SSM commands and access CloudWatch Logs

## Documentation 📚

For detailed documentation, please visit our [Wiki](../../wiki).

## Project Structure 🏗️

```
jobfit-analyzer/
├── app.py                 # Main Streamlit application (Docker version)
├── app_local.py           # Local development version
├── code.py                # Core functionality without GUI
├── Lambda_function.py     # AWS Lambda for S3 to EC2 sync
├── Dockerfile             # Docker configuration
├── .dockerignore          # Docker build exclusions
├── requirements.txt       # Project dependencies
```

## Contributing 🤝

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Roadmap 🛣️

- [ ] Add support for more document formats
- [ ] Implement batch processing
- [ ] Add advanced analytics dashboard
- [ ] Integrate with ATS systems
- [ ] Add multi-language support
- [ ] Enhance AWS deployment with CloudFormation templates
- [ ] Implement CI/CD pipeline

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- OpenAI for their powerful embedding models
- Streamlit team for the amazing framework
- AWS for cloud infrastructure solutions
- All our contributors and supporters

---

<div align="center">
Made with ❤️ by the JobFit Analyzer Team
</div>
