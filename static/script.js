document.getElementById("startInterview").addEventListener("click", function () {
    const role = document.getElementById("role").value;
    const type = document.getElementById("type").value;
    const experience = document.getElementById("experience").value;
    const skills = document.getElementById("skills").value.split(",").map(skill => skill.trim());

    if (!role || !type || !experience || skills.length === 0) {
        alert("Please fill out all fields before starting the interview.");
        return;
    }

    fetch("/start_recording", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            role: role,
            type: type,
            experience: experience,
            skills: skills
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Success:", data);
        document.getElementById("results").innerText = data.message;
    })
    .catch(error => console.error("Error starting interview:", error));
});


document.getElementById("stopInterview").addEventListener("click", function () {
    fetch("/stop_recording", { method: "POST" })
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById("results");
            resultsDiv.innerHTML = `<h2>Interview Results</h2><pre>${JSON.stringify(data, null, 2)}</pre>`;
            resultsDiv.style.display = "block";
        })
        .catch(error => console.error("Error stopping interview:", error));
});
