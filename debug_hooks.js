// PASTE VÀO CONSOLE TRƯỚC KHI ĐĂNG NHẬP
// Script này sẽ tự động log mọi thứ

console.log("🔍 Debug script loaded");

// Override localStorage.setItem để log
const originalSetItem = localStorage.setItem;
localStorage.setItem = function(key, value) {
  console.log(`📝 localStorage.setItem("${key}", "${value}")`);
  return originalSetItem.apply(this, arguments);
};

// Override localStorage.removeItem để log
const originalRemoveItem = localStorage.removeItem;
localStorage.removeItem = function(key) {
  console.log(`🗑️ localStorage.removeItem("${key}")`);
  return originalRemoveItem.apply(this, arguments);
};

// Log mọi fetch request
const originalFetch = window.fetch;
window.fetch = function(...args) {
  const url = args[0];
  if (url.includes('spotify')) {
    console.log(`🌐 fetch("${url}")`);
    return originalFetch.apply(this, arguments).then(r => {
      return r.clone().json().then(data => {
        console.log(`📥 Response:`, data);
        return r;
      }).catch(() => r);
    });
  }
  return originalFetch.apply(this, arguments);
};

console.log("✅ Debug hooks installed. Giờ bấm 'Đăng nhập Spotify'");
