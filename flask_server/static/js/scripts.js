// JavaScript code to handle user input and display chat
const chatContainer = document.getElementById('chat-container');
const submitButton = document.getElementById('submitButton');
const userInput = document.getElementById('message');
const image = document.getElementById('image-add-button');
const socket = io.connect('http://' + location.hostname + ':' + location.port);

submitButton.addEventListener('click', async (e) => {
    e.preventDefault();
    let imageSrc = null;

    const imageFile = image.files[0];
    if (imageFile) {
        imageSrc = await loadImage(imageFile);
        appendImage(imageSrc);
        if (image.files.length > 0) {
            // Clear the selected file
            image.value = '';
        }
    }
    if (userInput.value === '' && imageSrc === null) {
        return;
    }
    const userMessage = userInput.value;
    userInput.value = '';
    submitToSocket(userMessage, imageSrc);

    if (userMessage !== '') {
        appendMessage(userMessage);
    }
    await fetchBotMessage();
});

function submitToSocket(message, image) {
    socket.emit('submit', {message, image});
    console.log('submitted');
}

function loadImage(imageFile) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const imageSrc = event.target.result;
            resolve(imageSrc);
        };
        reader.readAsDataURL(imageFile);
    });
}

function appendMessage(message) {
    const messageElement = document.createElement('li');
    messageElement.className = 'user'
    messageElement.innerText = message
    chatContainer.appendChild(messageElement);
}

function appendImage(imageSrc) {
    const imageLi = document.createElement('li');
    imageLi.className = 'user';
    const imageElement = document.createElement('img');
    imageElement.src = imageSrc;
    imageElement.alt = `User image`;
    imageElement.width = 300;
    imageElement.height = 300;
    imageLi.appendChild(imageElement);
    chatContainer.appendChild(imageLi);
}

function createBotMessage() {
    const messageElement = document.createElement('li');
    messageElement.className = 'bot'
    messageElement.innerText = ''
    chatContainer.appendChild(messageElement);
    return messageElement;
}

function fetchBotMessage() {
    const botElement = createBotMessage();

    return new Promise((resolve) => {
        socket.on('content', function (data) {
            botElement.innerText.replaceAll('</s>', '')
            if (data === '//EOS//') {
                resolve();
                socket.off('content');
                return;
            }
            botElement.innerText += data;
            resolve();
        });
    });
}
document.getElementById('image-icon').addEventListener('click', function () {
    document.getElementById('image-add-button').click();
});
