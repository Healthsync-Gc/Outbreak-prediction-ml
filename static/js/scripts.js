document.getElementById('predictForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const periods = document.getElementById('periods').value;
    const frequency = document.getElementById('frequency').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ periods: periods, frequency: frequency }),
    })
    .then(response => response.json())
    .then(data => {
        let results = '<h2>Predictions</h2><ul>';
        data.forEach(item => {
            results += `<li>${item.ds}: ${item.yhat} (lower: ${item.yhat_lower}, upper: ${item.yhat_upper})</li>`;
        });
        results += '</ul>';
        document.getElementById('results').innerHTML = results;
    });
});
