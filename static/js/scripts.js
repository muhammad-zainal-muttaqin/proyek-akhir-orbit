/*!
 * Start Bootstrap - Landing Page v6.0.6 (https://startbootstrap.com/theme/landing-page)
 * Copyright 2013-2023 Start Bootstrap
 * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-landing-page/blob/master/LICENSE)
 */
// This file is intentionally blank
// Use this file to add JavaScript to your project
document.addEventListener("DOMContentLoaded", function () {
    var mybutton = document.getElementById("backToTop");

    window.onscroll = function () {
        if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
            mybutton.style.display = "block";
        } else {
            mybutton.style.display = "none";
        }
    };

    mybutton.addEventListener("click", function () {
        document.body.scrollTop = 0; // For Safari
        document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE, and Opera
    });
});