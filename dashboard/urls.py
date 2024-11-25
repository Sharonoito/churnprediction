from django.urls import path
from . import views

urlpatterns=[
    path('',views.dashboard_view, name='dashboard'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
]