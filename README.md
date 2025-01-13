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

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

## Overview 📋

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

## Quick Start 🚀

### Prerequisites

- Python 3.7+
- Git
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jobfit-analyzer.git
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

### Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Documentation 📚

For detailed documentation, please visit our [Wiki](../../wiki).

## Project Structure 🏗️

```
jobfit-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── src/
│   ├── analyzer/         # Core analysis modules
│   ├── utils/            # Utility functions
│   └── config/           # Configuration files
├── assets/               # Images and static files
└── tests/                # Test files
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

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- OpenAI for their powerful embedding models
- Streamlit team for the amazing framework
- All our contributors and supporters

---

<div align="center">
Made with ❤️ by the JobFit Analyzer Team
</div>
