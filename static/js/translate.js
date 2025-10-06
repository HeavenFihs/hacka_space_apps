// Translate dropdown
const translateIcon = document.querySelector('.translate-icon');
const dropdown = document.querySelector('.dropdown');
translateIcon.addEventListener('click', () => dropdown.classList.toggle('hidden'));

function setLang(lang) {
    if (lang === 'en') {
        document.cookie = "googtrans=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    } else {
        document.cookie = "googtrans=/en/" + lang + "; path=/";
    }
    location.reload();
}