{% extends 'basic.tpl'%}

{% block in_prompt %}
{% endblock in_prompt %}

{% block input %}
{% if 'HIDDEN' not in cell['source'] %}
<d-code language="python">
{{ cell['source'] }}
</d-code>
{% endif %}
{% endblock input %}


{% block markdowncell %}
{% if 'PROOF' not in cell['source'] %}
 {{ super() }}
 {% else %}
 <div class='proof_block'>
 <p> <a href="javascript:void(0);" class='proof_toggle'> Toggle proof </a> </p>
 <div class='proof'>
 {{ super() }}
 </div>
 </div>
{% endif %}
{% endblock markdowncell %}
