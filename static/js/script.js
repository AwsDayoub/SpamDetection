/*document.getElementById('predictForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var emailText = document.getElementById('email_text').value;
    var kValue = document.getElementById('kValue').value;
    fetch('/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
        },
        body: JSON.stringify({emailText: emailText, kValue: kValue}),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Prediction: ' + data.result;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
*/