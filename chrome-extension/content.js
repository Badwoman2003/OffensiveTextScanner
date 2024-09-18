function getAllText() {
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  let node;
  let textContent = [];

  while (node = walker.nextNode()) {
    const text = node.nodeValue.trim();
    if (text.length > 0) {
      textContent.push(text);
    }
  }

  return textContent;
}

function TransTextData() {
  const textContent = getAllText();
  alert(textContent.length)
  fetch('http://127.0.0.1:8080/App', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ data: textContent })
  })
    .then(response => response.json())
    .then(data => console.log('Success:', data))
    .catch(error => console.error('Error:', error));
}

function getAllImageUrls() {
  const images = document.querySelectorAll("img");
  let imageUrls = [];
  images.forEach((img) => {
    imageUrls.push(img.src);
  });
  return imageUrls;
}

TransTextData()