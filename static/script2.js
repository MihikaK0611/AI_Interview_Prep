document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById("resumeFile");
    formData.append("file", fileInput.files[0]);

    fetch("/analyze_resume", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("resultBox").textContent = JSON.stringify(data, null, 2);
    })
    .catch(err => {
        document.getElementById("resultBox").textContent = "Error analyzing resume.";
        console.error(err);
    });
});
