// OGP (Open Graph Protocol) and meta tags for social sharing
document.addEventListener("DOMContentLoaded", function() {
  // Get current page title and description
  const pageTitle = document.title;
  const pageDescription = document.querySelector('meta[name="description"]')?.content || 
                          "PyTorchを使った機械学習・深層学習の実践的なチュートリアル";
  const currentUrl = window.location.href;
  
  // Create meta tags for social sharing
  const metaTags = [
    // Open Graph
    { property: "og:type", content: "website" },
    { property: "og:title", content: pageTitle },
    { property: "og:description", content: pageDescription },
    { property: "og:url", content: currentUrl },
    { property: "og:site_name", content: "vinsmoke-three" },
    { property: "og:locale", content: "ja_JP" },
    
    // Twitter Card
    { name: "twitter:card", content: "summary_large_image" },
    { name: "twitter:title", content: pageTitle },
    { name: "twitter:description", content: pageDescription },
    { name: "twitter:url", content: currentUrl },
    
    // Additional meta
    { name: "author", content: "vinsmoke-three" },
    { name: "keywords", content: "PyTorch,機械学習,深層学習,Python,チュートリアル,日本語" },
    { name: "robots", content: "index,follow" }
  ];
  
  // Add meta tags to head
  metaTags.forEach(tag => {
    const existingTag = document.querySelector(`meta[${tag.property ? 'property' : 'name'}="${tag.property || tag.name}"]`);
    if (!existingTag) {
      const meta = document.createElement('meta');
      if (tag.property) {
        meta.setAttribute('property', tag.property);
      } else {
        meta.setAttribute('name', tag.name);
      }
      meta.setAttribute('content', tag.content);
      document.head.appendChild(meta);
    }
  });
});