
function handleResponse(data) {
    if (data.status === 0 && data.message === 'success') {
        const results = data.data;
        const offensiveTexts = Object.keys(results).filter(key => results[key] === 1);
        if (offensiveTexts.length > 0) {
            showWarning(offensiveTexts);
        } else {
            alert(offensiveTexts.length)
            showSafeMessage();
        }
    } else {
        console.error('Unexpected response format');
    }
}

function showWarning(offensiveTexts) {
    const warningDiv = document.createElement('div');
    warningDiv.style.position = 'fixed';
    warningDiv.style.top = '10px';
    warningDiv.style.right = '10px';
    warningDiv.style.padding = '20px';
    warningDiv.style.backgroundColor = 'red';
    warningDiv.style.color = 'white';
    warningDiv.style.zIndex = 1000;
    warningDiv.innerHTML = `
      <p>the page concludes offensive text</p>
      <button id="view-details">show details</button>
    `;
    document.body.appendChild(warningDiv);

    document.getElementById('view-details').addEventListener('click', () => {
        alert(`offensive text:\n\n${offensiveTexts.join('\n\n')}`);
    });
}

function showSafeMessage() {
    const safeDiv = document.createElement('div');
    safeDiv.style.position = 'fixed';
    safeDiv.style.top = '10px';
    safeDiv.style.right = '10px';
    safeDiv.style.padding = '20px';
    safeDiv.style.backgroundColor = 'green';
    safeDiv.style.color = 'white';
    safeDiv.style.zIndex = 1000;
    safeDiv.innerHTML = `<p>it's a safe page</p>`;
    document.body.appendChild(safeDiv);
}

function getResponse() {
    fetch('http://127.0.0.1:8080/get', {
        method: 'GET',
    })
        .then(response => response.json())
        .then(data => handleResponse(data))
        .catch((error) => console.error('Error:', error));
}

document.getElementById('scan-btn').addEventListener('click', function () {
    getResponse()
});