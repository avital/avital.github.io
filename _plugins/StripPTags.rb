module Jekyll
  module StripPTags
    def strip_p_tags(html)
      html.gsub(%r{\<\/?p.*?\>}, '')
    end
  end
end

Liquid::Template.register_filter(Jekyll::StripPTags)
