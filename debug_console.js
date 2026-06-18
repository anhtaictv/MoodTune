// PASTE ĐOẠN NÀY VÀO CONSOLE CỦA TRÌNH DUYỆT (F12 → Console tab)
// Sau khi paste, nó sẽ hiện thông tin debug

console.clear();
console.log("=== 🔍 SPOTIFY DEBUG ===");
console.log("");

// 1. Kiểm tra localStorage
const sid = localStorage.getItem("sp_sid");
console.log("1️⃣ localStorage sp_sid:", sid || "❌ EMPTY");

// 2. Kiểm tra URL params
const params = new URLSearchParams(window.location.search);
const urlSid = params.get("spotify_sid");
const urlErr = params.get("spotify_error");
console.log("2️⃣ URL param spotify_sid:", urlSid || "(none)");
console.log("3️⃣ URL param spotify_error:", urlErr || "(none)");

// 3. Kiểm tra biến global
console.log("4️⃣ Global spotifySid:", typeof spotifySid !== 'undefined' ? spotifySid : "❌ undefined");

// 4. Test API
if (sid) {
  console.log("");
  console.log("5️⃣ Testing API with stored sid...");
  fetch(`https://anhtaictv.me/api/spotify/status?sid=${sid}`)
    .then(r => r.json())
    .then(d => {
      console.log("   API response:", d);
      if (d.logged_in) {
        console.log("   ✅ Backend nhận ra token!");
      } else {
        console.log("   ❌ Backend KHÔNG nhận ra token - sid có thể đã hết hạn");
      }
    })
    .catch(e => console.error("   ❌ API error:", e));
} else {
  console.log("");
  console.log("5️⃣ Không có sid để test API");
}

// 5. Hướng dẫn
console.log("");
console.log("📋 HƯỚNG DẪN:");
console.log("- Nếu localStorage EMPTY: frontend không lưu được sid");
console.log("- Nếu có sid nhưng API trả logged_in=false: token đã hết hạn hoặc bị xóa");
console.log("- Nếu có sid và API trả logged_in=true: mọi thứ OK, có thể do frontend cache");
console.log("");
console.log("🔄 Để test lại: Bấm 'Đăng nhập Spotify', sau khi quay về paste lại đoạn này");
