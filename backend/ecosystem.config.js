module.exports = {
  apps: [
    {
      name: "moodtune-backend",
      script: "app.py",
      interpreter: "C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python314\\python.exe",
      cwd: "C:\\moodtune\\backend",
      env: {
        MOODTUNE_FRONTEND: "https://anhtaictv.me",
      },
      watch: false,
      autorestart: true,
    },
  ],
};
