async function sendValue() {
    const userInput = document.getElementById("userInput").value;
    const stockName = document.getElementById("stockName").value;

    const pattern = /^0\.\d+$/;
    if (!pattern.test(userInput)) {
        alert("Invalid input! Please enter a value in the format 0.x");
        return;
    }

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

    const resultHtml = await response.text();
    document.getElementById("result").innerHTML = resultHtml;
}
