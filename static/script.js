document
  .getElementById("startInterview")
  .addEventListener("click", function () {
    const role = document.getElementById("role").value;
    const type = document.getElementById("type").value;
    const experience = document.getElementById("experience").value;
    const skills = document
      .getElementById("skills")
      .value.split(",")
      .map((skill) => skill.trim());

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
        skills: skills,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Interview started...");
      })
      .catch((error) => console.error("Error starting interview:", error));
  });

document.getElementById("stopInterview").addEventListener("click", function () {
  fetch("/stop_recording", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      const resultsDiv = document.getElementById("results");
      resultsDiv.innerHTML = `
                <button id="generateReport">Generate Report</button>
            `;
      resultsDiv.style.display = "block";

      document
        .getElementById("generateReport")
        .addEventListener("click", function () {
          fetch("/generate_report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
          })
            .then((response) => response.text())
            .then((html) => {
              const newTab = window.open();
              newTab.document.open();
              newTab.document.write(html);
              newTab.document.close();
            })
            .catch((error) => console.error("Error generating report:", error));
        });
    })
    .catch((error) => console.error("Error stopping interview:", error));
});
