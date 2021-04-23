{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}
   :show-inheritance:


{% block attributes %}

{% if attributes %}
Properties
----------
   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}


      
{% block methods %}

{% if methods %}
Methods
-------
.. autosummary::
   :toctree: .
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}

Inheritence diagram
-------------------
    .. inheritance-diagram:: {{fullname}}
