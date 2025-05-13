document.getElementById('predictForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const year = document.getElementById('year').value;

  if (year < 1973 || year > 2099) {
    document.getElementById('result').innerText = 'Enter a year between 1973 and 2099.';
    return;
  }

  try {
  
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ Year: Number(year) }),
    });

    const data = await response.json();

    if (response.ok) {
      document.getElementById('result').innerText =
        `Predicted Gold Price for ${data.Year}: ${data['Predicted Gold Price']}`;
    } else {
      document.getElementById('result').innerText = `Error: ${data.error}`;
    }
  } catch (error) {
    document.getElementById('result').innerText = 'Error connecting to the server!';
  }
});
