<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <item value="-1.0" color="#d73027" label="-1.0 (No vegetation)" alpha="255"/>
          <item value="-0.5" color="#f46d43" label="-0.5" alpha="255"/>
          <item value="-0.2" color="#fdae61" label="-0.2" alpha="255"/>
          <item value="0.0"  color="#fee08b" label="0.0 (Bare)" alpha="255"/>
          <item value="0.1"  color="#ffffbf" label="0.1" alpha="255"/>
          <item value="0.2"  color="#d9ef8b" label="0.2 (Sparse)" alpha="255"/>
          <item value="0.4"  color="#a6d96a" label="0.4" alpha="255"/>
          <item value="0.6"  color="#66bd63" label="0.6 (Moderate)" alpha="255"/>
          <item value="0.8"  color="#1a9850" label="0.8 (Dense)" alpha="255"/>
          <item value="1.0"  color="#006837" label="1.0 (Very dense)" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
