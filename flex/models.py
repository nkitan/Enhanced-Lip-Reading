"""Flexible Page."""
from django.db import models

from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page
from wagtail.core.fields import StreamField

from streams import blocks

# Create your models here.

class FlexPage(Page):
    """Flexible page class"""
    template = "flex/flex_page.html"

    content = StreamField(
        [
            ("title_and_text", blocks.TitleAndTextBlock(classname='text_and_title')),
            ("full_rich_text", blocks.RichtextBlock()),
            ("simple_rich_text", blocks.SimpleRichtextBlock()),
            ("cards", blocks.CardBlock()),
        ],
        null=True,
        blank=True,
        use_json_field=True
    )


    content_panels = Page.content_panels + [

        FieldPanel("content"),
    ]

    class Meta:
        verbose_name = "Flex Page"
        verbose_name_plural = "Flex Pages"
