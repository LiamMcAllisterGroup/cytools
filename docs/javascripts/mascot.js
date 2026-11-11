// Restart the animation of the home page mascot every time the page is shown.
//
// Browsers keep a loaded SVG image, including the state of its animations, in
// memory and reuse it when the same URL is displayed again. Without this, the
// animation would only play the first time, and not after reloading the page or
// navigating back to it. Giving the image a unique URL makes the browser load a
// fresh copy, whose animation starts from the beginning.
document$.subscribe(() => {
  const image = document.querySelector(".cy-hero__image");
  if (image) {
    const url = new URL(image.src);
    url.searchParams.set("t", Date.now());
    image.src = url.href;
  }
});
