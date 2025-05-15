let currentQuestionIndex = 0;
let correctAnswers = 0;
let incorrectQuestions = [];
let answeredQuestions = [];
let totalQuestions = 0;
let initialTotalQuestions = 0; // Store the original total number of questions
let wordData = []; // Define wordData globally

async function fetchRandomWords() {
    try {
        const response = await fetch('/get_video_list'); // Endpoint to get video filenames
        const videoFiles = await response.json();
        
        if (videoFiles.length < 10) {
            console.error('Not enough video files available.');
            return;
        }

        shuffleArray(videoFiles);
        wordData = videoFiles.slice(0, 10).map(file => ({ word: file.replace('.mp4', '') }));
        
        initialTotalQuestions = wordData.length;
        totalQuestions = wordData.length;
        loadQuestion();
    } catch (error) {
        console.error('Error fetching video list:', error);
    }
}

// Shuffle function for randomizing questions
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Update progress bar after each correct answer
function updateProgressBar(forceComplete = false) {
    const progressBar = document.getElementById('quiz-progress');
    const progressText = document.getElementById('progress-text');

    // Calculate progress percentage
    const progressPercentage = forceComplete
        ? 100
        : (correctAnswers / initialTotalQuestions) * 100;

    progressBar.value = progressPercentage;
    progressText.textContent = `${Math.round(progressPercentage)}% completed`;
    
    // Update server with current progress after each answer
    updateServerProgress(progressPercentage);
}

// Function to update progress on the server
function updateServerProgress(progressPercentage) {
    fetch("/get_current_user")
        .then(response => response.json())
        .then(data => {
            if (data.user_id) {
                fetch("/update_progress", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({twoquiz: progressPercentage})
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.message) {
                        localStorage.setItem("twoquiz", progressPercentage);
                        console.log("Progress updated successfully:", progressPercentage);
                    } else {
                        console.error("Progress update error:", data.error || "Unknown error");
                    }
                })
                .catch(error => {
                    console.error("Error updating progress:", error);
                });
            } else {
                console.warn("User not logged in, progress not saved to server");
                // Still update local storage for guest users
                localStorage.setItem("twoquiz", progressPercentage);
            }
        })
        .catch(error => {
            console.error("Error getting user data:", error);
        });
}

function checkQuizCompletion() {
    let passingScore = 100; // 100% required to pass
    let progressPercentage = (correctAnswers / initialTotalQuestions) * 100;
    
    if (progressPercentage >= passingScore) {
        alert("Congratulations! You have completed Level 2.");
        completeLevel(2);
        localStorage.setItem("level2Completed", "true");
        window.location.href = "temp2.html"; // Redirect back to roadmap
    } else {
        alert("Try again! You need 100% to pass.");
        window.location.href = "word1.html";
    }
}

// Load question and display video
function loadQuestion() {
    const feedback = document.getElementById('feedback');
    const videoElement = document.getElementById('quiz-video');
    const optionsContainer = document.getElementById('options-container');

    if (currentQuestionIndex >= wordData.length) {
        feedback.textContent = 'Quiz completed!';
        checkQuizCompletion();
        updateProgressBar(true);
        return;
    }

    const question = wordData[currentQuestionIndex];

    // Ensure the video source is correctly set
    videoElement.src = `/static/videos/${question.word}.mp4`;
    videoElement.load();  // Reload the video
    videoElement.muted = true;
    videoElement.play();  // Auto-play

    // Clear previous options and feedback
    optionsContainer.innerHTML = '';
    feedback.textContent = '';

    // Generate randomized options
    const options = generateOptions(question.word);
    options.forEach((option, index) => {
        const button = document.createElement('button');
        button.classList.add('option');
        button.textContent = option;
        button.onclick = () => checkAnswer(index, question.word, options);
        optionsContainer.appendChild(button);
    });
}

function generateOptions(correctWord) {
    const options = new Set([correctWord]);
    while (options.size < 4) {
        const randomWord = wordData[Math.floor(Math.random() * wordData.length)].word;
        options.add(randomWord);
    }
    return Array.from(options).sort(() => Math.random() - 0.5);
}

// Check answer and provide feedback
function checkAnswer(selectedIndex, correctWord, options) {
    const feedback = document.getElementById('feedback');

    // If answer is correct
    if (options[selectedIndex] === correctWord) {
        correctAnswers++;
        feedback.textContent = 'Correct!';
        updateProgressBar();
    } else {
        feedback.textContent = `Incorrect! Correct answer: ${correctWord}`;
        // Add to incorrect list for potential review
        incorrectQuestions.push({
            word: correctWord,
            selectedAnswer: options[selectedIndex]
        });
    }

    // Add to answered questions
    answeredQuestions.push({
        word: correctWord,
        isCorrect: options[selectedIndex] === correctWord
    });

    // Proceed to next question
    currentQuestionIndex++;
    setTimeout(loadQuestion, 2000); // Wait 2 seconds before loading next question
}

function completeLevel(level) {
    fetch("/get_current_user")
        .then(response => response.json())
        .then(data => {
            if (data.user_id) {
                let newProgress = level + 1;
                let score = level + 1;
                let date = new Date().toISOString().split("T")[0];
                let score1 = (correctAnswers / initialTotalQuestions) * 100;
                
                // Remove the undefined update_user_streak call
                
                fetch("/update_progress", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        date: date, 
                        progress_level: newProgress, 
                        twoquiz: score1,
                        quiz: score
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok: ' + response.status);
                    }
                    return response.json();
                })
                .then(data => {
                    console.log("Update response data:", data);
                    if (data.message) {
                        localStorage.setItem("twoquiz", score1);
                        localStorage.setItem("quiz", score);
                        localStorage.setItem("progress_level", newProgress);
                        localStorage.setItem(`level${level}Completed`, "true");
                        alert("Level Completed! Next Level Unlocked.");
                        window.location.href = "roadmap.html";
                    } else {
                        alert("Error updating progress: " + (data.error || "Unknown error"));
                    }
                })
                .catch(error => {
                    console.error("Error updating progress:", error);
                    alert("There was a problem saving your progress. Please try again.");
                });
            } else {
                console.warn("User not logged in, redirecting to login");
                alert("Please log in first to save your progress.");
                window.location.href = "web.html";
            }
        })
        .catch(error => {
            console.error("Error getting user:", error);
        });
}

// Initialize quiz and start
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await fetchRandomWords();  // Ensure wordData is initialized first
        
        // Check if these are valid after data is loaded
        if (wordData && wordData.length > 0) {
            initialTotalQuestions = wordData.length; 
            totalQuestions = wordData.length;  
            shuffleArray(wordData); 
            loadQuestion();
        } else {
            throw new Error("Failed to load quiz data");
        }
    } catch (error) {
        console.error("Error initializing quiz:", error);
        alert("There was a problem loading the quiz. Please refresh the page or try again later.");
    }
});