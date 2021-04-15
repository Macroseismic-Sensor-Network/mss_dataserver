{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}
.. automethod:: {{ objname }}

{% block attributes %}

{% print members %}

{% if attributes %}
Attributes
----------

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}
