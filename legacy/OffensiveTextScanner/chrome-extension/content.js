function getAllText() {
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );

  let node;
  let textContent = [];

  while ((node = walker.nextNode())) {
    // 跳过 <script> 标签内的文本
    if (node.parentNode && node.parentNode.nodeName.toLowerCase() === 'script') {
      continue;
    }

    const text = node.nodeValue.trim();
    if (text.length > 0) {
      textContent.push(text);
    }
  }

  return textContent;
}

function getAllImageUrls() {
  const images = document.querySelectorAll("img");
  let imageUrls = [];
  images.forEach((img) => {
    imageUrls.push(img.src);
  });
  return imageUrls;
}

function TransPageData() {
  window.onload = function () {
    // 提取页面文本数据
    const textContent = getAllText();
    
    // 提取页面图像数据
    const ImgContent = getAllImageUrls();
    
    // 弹窗显示提取到的文本数量
    alert(textContent.length);

    // 发送数据到指定服务器
    fetch("http://127.0.0.1:8080/App", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ TextData: textContent, ImgData: ImgContent }),
    })
      .then((response) => response.json())
      .then((data) => console.log("Success:", data))
      .catch((error) => console.error("Error:", error));
  };
}

TransPageData();
