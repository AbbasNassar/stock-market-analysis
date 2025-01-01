async function sendValue() {
    const userInput = document.getElementById("userInput").value;
    const stockName = document.getElementById("stockName").value;

    // Validate input to match the 0.x pattern
    const pattern = /^0\.\d+$/;
    if (!pattern.test(userInput)) {
        alert("Invalid input! Please enter a value in the format 0.x");
        return; // Stop execution if validation fails
    }

    // If validation passes, proceed with the fetch request
    const response = await fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            value: userInput,
            stockName: stockName
        }),
    });

    // Parse the HTML response and display it
    const resultHtml = await response.text();
    document.getElementById("result").innerHTML = resultHtml;
}
