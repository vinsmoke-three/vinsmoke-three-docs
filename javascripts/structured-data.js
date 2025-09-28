// Add structured data (JSON-LD) for better SEO
document.addEventListener("DOMContentLoaded", function() {
  // Basic website structured data
  const websiteSchema = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "vinsmoke-three",
    "url": "https://vinsmoke-three.com",
    "description": "PyTorchを使った機械学習・深層学習の実践的なチュートリアル",
    "inLanguage": "ja",
    "author": {
      "@type": "Person",
      "name": "vinsmoke-three"
    },
    "potentialAction": {
      "@type": "SearchAction",
      "target": "https://vinsmoke-three.com/search?q={search_term_string}",
      "query-input": "required name=search_term_string"
    }
  };

  // Educational organization schema
  const organizationSchema = {
    "@context": "https://schema.org",
    "@type": "EducationalOrganization",
    "name": "vinsmoke-three",
    "url": "https://vinsmoke-three.com",
    "description": "機械学習・深層学習の教育リソースを提供",
    "educationalCredentialAwarded": "実践的な機械学習スキル",
    "hasCredential": {
      "@type": "EducationalOccupationalCredential",
      "credentialCategory": "技術スキル",
      "competencyRequired": "PyTorch, 機械学習, 深層学習"
    }
  };

  // Check if we're on a tutorial page
  const currentPath = window.location.pathname;
  if (currentPath.includes('/PyTorch/')) {
    const pageTitle = document.title;
    const pageDescription = document.querySelector('meta[name="description"]')?.content;
    
    const tutorialSchema = {
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": pageTitle,
      "description": pageDescription,
      "url": window.location.href,
      "datePublished": document.querySelector('meta[name="date"]')?.content || "2025-01-01",
      "dateModified": new Date().toISOString().split('T')[0],
      "author": {
        "@type": "Person",
        "name": "vinsmoke-three"
      },
      "publisher": {
        "@type": "Organization",
        "name": "vinsmoke-three",
        "url": "https://vinsmoke-three.com"
      },
      "mainEntityOfPage": {
        "@type": "WebPage",
        "@id": window.location.href
      },
      "inLanguage": "ja",
      "about": {
        "@type": "Thing",
        "name": "機械学習",
        "sameAs": "https://ja.wikipedia.org/wiki/機械学習"
      },
      "teaches": ["PyTorch", "機械学習", "深層学習", "Python"],
      "educationalLevel": "中級",
      "learningResourceType": "チュートリアル"
    };

    addStructuredData(tutorialSchema);
  }

  // Add basic schemas
  addStructuredData(websiteSchema);
  addStructuredData(organizationSchema);

  function addStructuredData(schema) {
    const script = document.createElement('script');
    script.type = 'application/ld+json';
    script.textContent = JSON.stringify(schema);
    document.head.appendChild(script);
  }
});