from django.template import RequestContext

from django.template.loader import get_template

return render_to_response("my_app/base.html", {'some_var': 'foo'},
                           context_instance=RequestContext(request))



template = get_template('hello.html')
html = template.render({'name': 'world'}, request)