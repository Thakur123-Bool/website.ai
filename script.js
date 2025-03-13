const uploadForm = document.getElementById('upload-form');
const fileUpload = document.getElementById('file-upload');
const description = document.getElementById('description');
const uploadStatus = document.getElementById('upload-status');
const askBtn = document.getElementById('ask-btn');
const questionInput = document.getElementById('question');
const responseDiv = document.getElementById('response');

// Upload PDF file and description to FastAPI
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    uploadStatus.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('file', fileUpload.files[0]);
    formData.append('description', description.value);

    try {
        const response = await fetch('https://dear-squid-techpet-13ad4475.koyeb.app/upload_documents/', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        uploadStatus.textContent = data.status || 'File uploaded successfully! Ask your questions now!';
    } catch (error) {
        uploadStatus.textContent = 'Error uploading file: ' + error.message;
    }
});

// Ask a question and get a response from FastAPI
askBtn.addEventListener('click', async () => {
    const question = questionInput.value;
    if (!question) {
        alert('Please enter a question.');
        return;
    }

    responseDiv.innerHTML = 'Loading...';

    try {
        const response = await fetch('https://dear-squid-techpet-13ad4475.koyeb.app/ask_question/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        responseDiv.innerHTML = data.response || 'No answer available.';
    } catch (error) {
        responseDiv.innerHTML = 'Error getting response: ' + error.message;
    }
});
document.getElementById('submit-button').addEventListener('click', async function() {
    const data = {
        name: document.getElementById('name').value,
        message: document.getElementById('message').value
    };

    try {
        const response = await fetch('https://dear-squid-techpet-13ad4475.koyeb.app/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        console.log('Response:', result);
    } catch (error) {
        console.error('Error:', error);
    }
});
